import torch
from torch import Tensor
from typing import Callable, Optional
_OptionalCallable = Optional[Callable]


@torch.no_grad()
def linear_solve(Cops: Callable, y: Tensor, x_init: Optional[Tensor] = None,
                 preconditioner: _OptionalCallable = None,
                 n_iter: int = 10, eps: float = 1e-7, verbose: bool = False):
    '''
    Conjugate Gradient method to solve a linear system Cx = y, where 
    C is a symmetric, positive definite matrix


    Args:
        - Cops [Callable]: the operator C
        - y [Tensor]: of shape (B, C, H', W')  a batch of images
        - x_init [Optional]: initial guess of the solutions

    Returns:
        - x [Tensor] the solution of the linear system, of shape (B, C, H, W)
    '''
    if preconditioner is None:
        def preconditioner(x): return x
    if x_init is None:
        x_init = preconditioner(y)

    x = x_init
    r = y - Cops(x)
    p = r.clone()
    r_new = r.clone()

    # Loop
    for k in range(n_iter):
        r = r_new.clone()
        Ap = Cops(p)
        r_norm = (r * r).sum(dim=(-1, -2), keepdim=True)
        alpha = r_norm / ((p * Ap).sum(dim=(-1, -2), keepdim=True))
        x = x + alpha * p
        r_new = r - alpha * Ap

        r_norm_new = (r_new * r_new).sum(dim=(-1, -2), keepdim=True)
        if torch.all(r_norm_new < eps):    # torch.all or torch.any
            if verbose:
                print(
                    "Conjugate gradient for solving linear system converged at iteration ", k + 1)
            return x

        beta = r_norm_new / r_norm
        p = r_new + beta * p

    if verbose:
        print("Conjugate gradient for solving linear system ended at iteration ", k + 1)

    return x 
