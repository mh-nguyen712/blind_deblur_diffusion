import torch
from .convolution import conv2d, conv2d_transpose
from functools import partial
from .linalg import linear_solve
import deepinv as dinv
from tqdm import tqdm
import os
import numpy as np

def _grad_kernel_fn(x, kernel, y, padding='valid'):
    def l2(x, kernel, y):
        Ax = conv2d(x, kernel, padding=padding)
        diff = Ax - y
        norm = 0.5 * torch.linalg.norm(diff.view(diff.size(0), -1), dim = 1) ** 2 
        return norm.sum()
    return torch.func.grad(l2, argnums=1)(x, kernel, y)

def pad(x, pad):
    return torch.nn.functional.pad(x, (pad, pad, pad, pad))

def grad_kernel(x, kernel, y, padding='valid'):
    if x.ndim == 4:
        x = x[:,None]
    if y.ndim == 4:
        y = y[:, None]
    if kernel.ndim == 4:
        kernel = kernel[:, None]
    
    kernel_size = kernel.size(-1)
    y = y[..., kernel_size // 2 : - kernel_size // 2 + 1, kernel_size // 2 : - kernel_size // 2 + 1]
    
    return torch.vmap(partial(_grad_kernel_fn, padding=padding), in_dims=(0,0,0))(x, kernel, y).squeeze(1)

@torch.no_grad()
def data_fidelity(x, kernel, y, padding='valid'):
    kernel_size = kernel.size(-1)
    y = y[..., kernel_size // 2 : - kernel_size // 2 + 1, kernel_size // 2 : - kernel_size // 2 + 1]
    
    A = partial(conv2d, kernel=kernel, padding=padding)
    return 0.5 * torch.norm((A(x) - y).view(x.size(0), -1), dim=-1) ** 2

@torch.no_grad()
def map_grad_x(x, kernel, y, grad_potential_fn, lamb = 1., padding='valid'):
    kernel_size = kernel.size(-1)
    y = y[..., kernel_size // 2 : - kernel_size // 2 + 1, kernel_size // 2 : - kernel_size // 2 + 1]
    
    A = partial(conv2d, kernel=kernel, padding=padding)
    A_adjoint = partial(conv2d_transpose, kernel=kernel,  padding=0 if padding=='valid' else kernel_size // 2) #kernel.size(-1) // 2)
    
    return A_adjoint(A(x) - y) - lamb * pad(grad_potential_fn(x[..., kernel_size // 2 : - kernel_size // 2 + 1, kernel_size // 2 : - kernel_size // 2 + 1]), kernel_size // 2) 



@torch.no_grad()
def projection_simplex_sort(v, z=1):
    shape = v.shape
    B = shape[0]
    v = v.view(B, -1)
    n_features = v.size(1)
    u = torch.sort(v, descending=True, dim = -1).values
    cssv = torch.cumsum(u, dim = -1) - z
    ind = torch.arange(n_features, device=v.device)[None,:].expand(B, -1) + 1.
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = torch.maximum(v - theta, torch.zeros_like(v))
    return w.reshape(shape)

@torch.no_grad()
def prox_l2(kernel, y, z, gamma, padding='valid'):
    kernel_size = kernel.size(-1)
    y = y[..., kernel_size // 2 : - kernel_size // 2 + 1, kernel_size // 2 : - kernel_size // 2 + 1]

    A = partial(conv2d, kernel=kernel, padding=padding)
    A_adjoint = partial(conv2d_transpose, kernel=kernel, padding = 0 if padding=='valid' else kernel_size // 2)
    
    Cops = lambda x: A_adjoint(A(x)) + x / gamma
    b = A_adjoint(y) + z / gamma 

    return linear_solve(Cops, b, n_iter=1000, eps=1e-5, verbose=False)


@torch.no_grad()
def optimization_loop(y, 
                      kernel_init,
                      denoiser, 
                      outer_loop: int = 10, 
                      max_num_steps: int = 50, 
                      min_num_steps : int = 10,
                      lr_x : float = 5e-2,
                      lr_kernel: float = 1e-3,
                      n_iter: int = 10, 
                      lamb: float = 1.,
                      save_dir: str = None,
                      num_warmup_steps = 15,
                      display_freq = 100,
                      lr_x_decay : float = 0.999
                      ):
    
    # --------------------------------------------
    x_init = y.clone()
    x_hat = x_init.detach().clone().requires_grad_(False)
    kernel_hat = kernel_init.clone().expand(y.size(0), -1, -1, -1)
    kernel_size = kernel_init.size(-1)
    # --------------------------------------------
    kernel_evolution = [kernel_init]
    x_evolution = [x_init]

    # --------------------------------------------
    b = y.size(0)
    delta = 0.2
    global_iteration = 0
    display_iteration = [0]
    if isinstance(lamb, float):
        lamb = [lamb] * outer_loop
    scale_dr = 0.03
    
    # -------------- optimizer --------------------

    lr_theta_percents = np.logspace(-2, -7, outer_loop * n_iter)

    os.makedirs(save_dir, exist_ok = True)
    old_save_path = ""
    print("Starting optimization loop...")
    def crop(x):
        return x[..., kernel_size // 2: -kernel_size // 2 + 1,kernel_size // 2: -kernel_size // 2 + 1]
    
    for outer in range(outer_loop):
        print("-" * 100)
        print("\t Outer iteration:", outer + 1, "/", outer_loop)
        steps = np.maximum(max_num_steps - np.arange(n_iter) * (max_num_steps - min_num_steps) / num_warmup_steps, min_num_steps).astype(np.int16)
        csteps = np.cumsum(steps)
        display_list = np.unique(np.searchsorted(csteps, np.arange(1, csteps[-1], display_freq))).tolist()
        t = tqdm(range(n_iter))
        x_hat = y.clone()
        lr_x *= lr_x_decay
                
        for i in t:
            n_iter_inner_x = steps[i]
            z = x_hat
            for k in range(n_iter_inner_x):
                x_hat = denoiser(z, lr_x * scale_dr)  # prox_tau_g
                z = z + prox_l2(kernel_hat, y, 2 * x_hat - z, lr_x) - x_hat
                z = torch.clamp(z, -delta, delta + 1)
            x_hat = denoiser(z, lr_x * scale_dr)
            
            # Gradient step on theta
            grad_theta = grad_kernel(x_hat, kernel_hat, y)
            lr_theta_percent = lr_theta_percents[global_iteration]
            lr_theta = max(min(lr_theta_percent / grad_theta.abs().max() * kernel_hat.abs().max(), lr_kernel), 1e-7)
            kernel_hat = kernel_hat - lr_theta * grad_theta
            kernel_hat = projection_simplex_sort(kernel_hat, z=1)
            
            l2 = data_fidelity(x_hat, kernel_hat, y).mean()
            t.set_description(f'grad: {grad_theta.norm():.3f}, {lr_theta:.3e}, l2 = {l2:.2f}')
            
            if i in display_list:
                x_evolution.append(x_hat)
                kernel_evolution.append(kernel_hat)
                display_iteration.append(global_iteration)

                dinv.utils.plot([crop(x_evolution[-1]), crop(y), kernel_evolution[-1]], 
                                titles=["Rec. image", "Blurry", "Rec. kernel"],
                                suptitle=f'Iteration {global_iteration:05d}',
                                figsize=(10, 3),
                                save_fn=os.path.join(save_dir, f'iteration_{global_iteration:05d}.png')
                          )


            global_iteration += 1

        _kernel_evolution_tensor = torch.stack(kernel_evolution, dim = 0)
        _x_evolution_tensor = torch.stack(x_evolution, dim = 0)

        if os.path.exists(old_save_path):
            os.remove(old_save_path)
        save_path = os.path.join(save_dir, f'result_iteration_{global_iteration}.pt')
        torch.save(dict(kernel_evolution=_kernel_evolution_tensor,
                       x_evolution=_x_evolution_tensor,
                       ),
                       save_path
                  )
        old_save_path = save_path

    return _x_evolution_tensor, _kernel_evolution_tensor

