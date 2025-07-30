
import torch
import torch.nn as nn
import numpy as np
from .utils import get_edm_parameters
from scipy import integrate
from typing import Callable
from torch import Tensor

class ScorePrior(nn.Module):
    def __init__(self, denoiser, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoiser = denoiser

    def grad(self, x: torch.Tensor, sigma_denoiser, *args, **kwargs):
        return (1 / sigma_denoiser**2) * (
            self.denoiser(x, sigma_denoiser, *args, **kwargs) - x
        )

    def score(self, x: torch.Tensor, sigma_denoiser, *args, **kwargs):
        return (1 / sigma_denoiser**2) * (
            self.denoiser(x, sigma_denoiser, *args, **kwargs) - x
        )
    def scaled_score(self, x: torch.Tensor, sigma_denoiser, *args, **kwargs):
        return self.denoiser(x, sigma_denoiser, *args, **kwargs) - x 


def get_ode_flow(prior, name="edm"):
    params = get_edm_parameters(name)
    timesteps_fn = params["timesteps_fn"]
    sigma_fn = params["sigma_fn"]
    sigma_deriv_fn = params["sigma_deriv_fn"]
    s_fn = params["s_fn"]
    s_deriv_fn = params["s_deriv_fn"]

    def ode_flow(x, t, *args, **kwargs):
        return s_deriv_fn(t) / s_fn(t) * x - s_fn(t) ** 2 * sigma_deriv_fn(
            t
        ) * sigma_fn(t) * prior.score(x / s_fn(t), sigma_fn(t), *args, **kwargs)

    t_spans = timesteps_fn(2)
    return ode_flow, t_spans


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_div_fn(fn: Callable):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn


@torch.no_grad()
def likelihood_fn(
    data: Tensor,
    ode_flow: Callable,
    t_spans=[1, 1e-3],
    hutchinson_type: str = "Rademacher",
    rtol=1e-5,
    atol=1e-5,
    method="RK45",
    *args,
    **kwargs,
):
    sigma_max = t_spans[0]

    def prior_logp(z):
        shape = z.shape
        n_time_steps = np.prod(shape[1:])
        return -n_time_steps / 2.0 * np.log(2 * np.pi * sigma_max**2) - torch.sum(
            z**2, dim=(1, 2, 3)
        ) / (2 * sigma_max**2)

    def div_fn(x, t, noise):
        """
        Approximates the divergence of the drift function of the reverse SDE
        i.e., evaluate epsilon^T \nabla f(x, t) epsilon in eq (40)
        """
        return get_div_fn(lambda xx, tt: ode_flow(xx, tt, *args, **kwargs))(x, t, noise)

    shape = data.shape
    if hutchinson_type.lower() == "gaussian":
        epsilon = torch.randn_like(data)
    elif hutchinson_type.lower() == "rademacher":
        epsilon = torch.randint_like(data, low=0, high=2).type(data.dtype) * 2 - 1.0
    else:
        raise NotImplementedError(
            "Hutchinson type %s is not implemented." % hutchinson_type
        )

    def ode_func(t, x):
        sample = (
            from_flattened_numpy(x[: -shape[0]], shape)
            .to(data.device)
            .type(torch.float32)
        )
        drift = to_flattened_numpy(ode_flow(sample, t, *args, **kwargs))
        logp_grad = to_flattened_numpy(div_fn(sample, t, epsilon))
        return np.concatenate([drift, logp_grad], axis=0)

    init = np.concatenate([to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
    solution = integrate.solve_ivp(
        ode_func, tuple(t_spans[::-1]), init, rtol=rtol, atol=atol, method=method
    )
    zp = solution.y[:, -1]
    z = from_flattened_numpy(zp[: -shape[0]], shape).to(data.device).type(torch.float32)
    delta_logp = (
        from_flattened_numpy(zp[-shape[0] :], (shape[0],))
        .to(data.device)
        .type(torch.float32)
    )
    prior_logpx = prior_logp(z)
    nll = -(prior_logpx + delta_logp)
    return nll


def ode_sampler(
    ode_flow: Callable,
    shape,
    t_spans,
    rtol=1e-5,
    atol=1e-5,
    method="RK45",
    device='cuda',
    *args,
    **kwargs,
):

    def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        drift = ode_flow(
            x,
            t,
            *args,
            **kwargs,
        )
        return to_flattened_numpy(drift)

    noise = torch.randn(shape, device=device, dtype=torch.float32) * t_spans[0]
    solution = integrate.solve_ivp(
        ode_func,
        tuple(t_spans.tolist()),
        to_flattened_numpy(noise),
        rtol=rtol,
        atol=atol,
        method=method,
    )
    nfe = solution.nfev
    ode_sample = (
        torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
    )

    return ode_sample, nfe
  
