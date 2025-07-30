"""
Computation of the likelihood using ODE solver
Modified from: https://github.com/yang-song/score_sde_pytorch/
"""

import torch
import numpy as np
from scipy import integrate
from .models import utils as mutils
# from torchdiffeq import odeint
from typing import Callable
import gc


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


def get_likelihood_fn(
    sde,
    hutchinson_type="Rademacher",
    rtol=1e-5,
    atol=1e-5,
    method="RK45",
    eps=1e-5,
    backends="scipy",
):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point.

    Args:
        sde: A `sde_lib.SDE` object that represents the forward SDE.
        inverse_scaler: The inverse data normalizer.
        hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
        rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
        atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
        method: A `str`. The algorithm for the black-box ODE solver.
            See documentation for `scipy.integrate.solve_ivp`.
        eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.
        backends: using 'scipy.integrate.solve_ivp' or 'torchdiffeq.odeint'

    Returns:
        A function that a batch of data points and returns the log-likelihoods in bits/dim,
            the latent code, and the number of function evaluations cost by computation.
    """

    def drift_fn(model, x, t):
        """
        The drift function of the reverse-time SDE.
        in eq. (6) or (13) of https://arxiv.org/abs/2011.13456, depending on weather we use prob flow or not
        """
        score_fn = mutils.get_score_fn(
            sde, model, train=False, continuous=True)
        # Probability flow ODE is a special case of Reverse SDE
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def div_fn(model, x, t, noise):
        """
        Approximates the divergence of the drift function of the reverse SDE
        i.e., evaluate epsilon^T \nabla f(x, t) epsilon in eq (40)
        """
        return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)

    def likelihood_fn(model, data):
        gc.collect()

        with torch.no_grad():
            shape = data.shape
            if hutchinson_type.lower() == "gaussian":
                epsilon = torch.randn_like(data)
            elif hutchinson_type.lower() == "rademacher":
                epsilon = (
                    torch.randint_like(data, low=0, high=2).type(
                        data.dtype) * 2 - 1.0
                )
            else:
                raise NotImplementedError(
                    "Hutchinson type %s is not implemented." % hutchinson_type
                )


            def ode_func(t, x):
                sample = mutils.from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
                vec_t = torch.ones(sample.shape[0], device=sample.device) * t
                drift = mutils.to_flattened_numpy(drift_fn(model, sample, vec_t))
                logp_grad = mutils.to_flattened_numpy(div_fn(model, sample, vec_t, epsilon))
                return np.concatenate([drift, logp_grad], axis=0)

            init = np.concatenate([mutils.to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
            solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)

            # nfe = solution.nfev
            zp = solution.y[:, -1]
            z = mutils.from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
            delta_logp = mutils.from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
    
            if hasattr(sde, "is_augmented"):
                if sde.is_augmented:
                    prior_logpx, prior_logpz = sde.prior_logp(z)
                    nll = -(prior_logpx + prior_logpz + delta_logp)
            else:
                prior_logpx = sde.prior_logp(z)
                nll = -(prior_logpx + delta_logp)

            return nll  

    return likelihood_fn
