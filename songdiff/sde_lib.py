"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.
Modified from: https://github.com/yang-song/score_sde_pytorch/
"""

import abc
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from .sde_utils import is_one_dim_tensor


class SDE(abc.ABC):
    """SDE abstract class.
    Functions are designed for a mini-batch of inputs of images: (B, C, H, W)
    """

    def __init__(self, n_time_steps: int) -> None:
        """Construct an SDE.

        Args:
        n_time_steps: number of discretization time steps.
        """
        super().__init__()
        self.n_time_steps = n_time_steps

    @property
    @abc.abstractmethod
    def T(self) -> float:
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x: Tensor = None, t: float = 0.0):
        """Drift function and diffusion coefficient which determine the forward SDE"""
        pass

    @abc.abstractmethod
    def marginal_prob(self, x: Tensor = None, t: float = 0.0):
        """Parameters (e.g., mean and std) to determine the marginal distribution of the SDE, $p_t(x_t)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x_T)$ (e.g., Normal or Uniform)."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
        z: latent code
        Returns:
        log probability density
        """
        pass

    def discretize(self, x: Tensor = None, t: float = 0.0):
        """
        Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probability flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
        x: a torch tensor
        t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f: discretized drift function, i.e. f(x_t, t) dt, tensor of size (1, None, None, None)
            G: discretized diffusion coefficient, i.e., g(t)sqrt(dt) a tensor of shape (1) (a scalar)
        """
        dt = self.T / self.n_time_steps
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=x.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
        score_fn: A time-dependent score-based model that takes x and t and returns the score.
        probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.

        Returns: instance of Reverse SDE class
        """
        n_time_steps = self.n_time_steps
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.n_time_steps = n_time_steps
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x: Tensor = None, t: float = 0.0):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""

                # drift and diffusion coefficient of the forward process
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)

                # drift and diffusion coefficient of the reverse process
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (
                    0.5 if self.probability_flow else 1.0
                )
                # Set the diffusion function to zero for ODEs.
                diffusion = 0.0 if self.probability_flow else diffusion

                return drift, diffusion

            def discretize(self, x: Tensor = None, t: float = None):
                """
                Create discretized iteration rules for the reverse diffusion sampler.
                Returns: discretized version of drift and diffusion of the reverse SDE, i.e., (f(x_t, t) - g^2(t) score(x_t))dt and g(t)sqrt(dt), dt = T / n_time_steps
                """

                # Get discretized version: f(x_t,t)dt and g(t)dt (of the forward SDE)
                f, G = discretize_fn(x, t)

                # Get discretized version of drift and diffusion of reverse SDE
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (
                    0.5 if self.probability_flow else 1.0
                )
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, n_time_steps=1000):
        """Construct a Variance Preserving SDE.

        Args:
        beta_min: value of beta(0)
        beta_max: value of beta(1)
        n_time_steps: number of discretization steps
        """
        super().__init__(n_time_steps)
        self.beta_0 = (
            beta_min if is_one_dim_tensor(beta_min) else torch.tensor([beta_min])
        )
        self.beta_1 = (
            beta_max if is_one_dim_tensor(beta_max) else torch.tensor([beta_max])
        )
        self.n_time_steps = n_time_steps
        self.discrete_betas = torch.linspace(
            beta_min / n_time_steps, beta_max / n_time_steps, n_time_steps
        )
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self):
        return 1.0

    def sde(self, x: Tensor = None, t: float = 0.0):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x: Tensor = None, t: float = 0.0):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        """
        Sampling from the prior distribution, i.e. Normal distribution
        """
        return torch.randn(*shape)

    def prior_logp(self, z):
        """
        log density of the prior density, i.e. log of the density of the normal distribution
        """
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0
        return logps

    def discretize(self, x: Tensor = None, t: float = 0.0):
        """
        DDPM discretization.
        """
        timestep = t * (self.n_time_steps - 1.0) / self.T
        if isinstance(t, Tensor):
            timestep = timestep.int()
        else:
            timestep = int(timestep)

        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G

    def to_device(self, device: str = "cuda"):
        for property, value in vars(self).items():
            if isinstance(value, Tensor):
                setattr(self, property, value.to(device))


class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, n_time_steps=1000):
        """Construct the sub-VP SDE that excels at likelihoods.

        Args:
        beta_min: value of beta(0)
        beta_max: value of beta(1)
        n_time_steps: number of discretization steps
        """
        super().__init__(n_time_steps)
        self.beta_0 = (
            beta_min if is_one_dim_tensor(beta_min) else torch.tensor([beta_min])
        )
        self.beta_1 = (
            beta_max if is_one_dim_tensor(beta_max) else torch.tensor([beta_max])
        )
        self.n_time_steps = n_time_steps

    @property
    def T(self):
        return 1.0

    def sde(self, x: Tensor = None, t: float = 0.0):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        discount = 1.0 - torch.exp(
            -2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2
        )
        diffusion = torch.sqrt(beta_t * discount)
        return drift, diffusion

    def marginal_prob(self, x: Tensor = None, t: float = 0.0):
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
        std = 1 - torch.exp(2.0 * log_mean_coeff)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        n_time_steps = np.prod(shape[1:])
        return (
            -n_time_steps / 2.0 * np.log(2 * np.pi)
            - torch.sum(z**2, dim=(1, 2, 3)) / 2.0
        )


class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, n_time_steps=1000):
        """Construct a Variance Exploding SDE.

        Args:
        sigma_min: smallest sigma.
        sigma_max: largest sigma.
        n_time_steps: number of discretization steps
        """
        super().__init__(n_time_steps)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Discrete sigma: sigma(t) = sigma_min * (sigma_max / sigma_min)^t
        self.discrete_sigmas = torch.exp(
            torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), n_time_steps)
        )
        self.n_time_steps = n_time_steps

    @property
    def T(self):
        return 1

    def sde(self, x: Tensor = None, t: float = 0.0):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(
            torch.tensor(
                2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device
            )
        )
        return drift, diffusion

    def marginal_prob(self, x: Tensor = None, t: float = 0.0):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        n_time_steps = np.prod(shape[1:])
        return -n_time_steps / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - torch.sum(z**2, dim=(1, 2, 3)) / (2 * self.sigma_max**2)

    def discretize(self, x: Tensor = None, t: float = 0.0):
        """
        SMLD (NCSN) discretization.
        """

        timestep = t * (self.n_time_steps - 1.0) / self.T
        if isinstance(t, Tensor):
            timestep = timestep.int()
        else:
            timestep = int(timestep)

        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent_sigma = torch.where(
            timestep == 0,
            torch.zeros_like(t),
            self.discrete_sigmas.to(t.device)[timestep - 1],
        )
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma**2 - adjacent_sigma**2)
        return f, G


class CLD(nn.Module):
    r"""
        Score-Based Generative Modeling
    with Critically-Damped Langevin Diffusion. ICLR 2022 Spotlight
    """

    def __init__(self, config, beta_fn, beta_int_fn):
        super().__init__()
        self.config = config
        self.beta_fn = beta_fn
        self.beta_int_fn = beta_int_fn
        self.m_inv = config.m_inv
        self.f = 2.0 / np.sqrt(config.m_inv)
        self.g = 1.0 / self.f
        self.gamma = config.gamma
        self.numerical_eps = config.numerical_eps

    @property
    def type(self):
        return "cld"

    @property
    def is_augmented(self):
        return True

    def sde(self, u, t):
        """
        Evaluating drift and diffusion of the SDE.
        """
        x, v = torch.chunk(u, 2, dim=1)

        beta = add_dimensions(self.beta_fn(t), self.config.is_image)

        drift_x = self.m_inv * beta * v
        drift_v = -beta * x - self.f * self.m_inv * beta * v

        diffusion_x = torch.zeros_like(x)
        diffusion_v = torch.sqrt(2.0 * self.f * beta) * torch.ones_like(v)

        return torch.cat((drift_x, drift_v), dim=1), torch.cat(
            (diffusion_x, diffusion_v), dim=1
        )

    def get_reverse_sde(self, score_fn=None, probability_flow=False):
        sde_fn = self.sde

        def reverse_sde(u, t, score=None):
            """
            Evaluating drift and diffusion of the ReverseSDE.
            """
            drift, diffusion = sde_fn(u, 1.0 - t)
            score = score if score is not None else score_fn(u, 1.0 - t)

            drift_x, drift_v = torch.chunk(drift, 2, dim=1)
            _, diffusion_v = torch.chunk(diffusion, 2, dim=1)

            reverse_drift_x = -drift_x
            reverse_drift_v = -drift_v + diffusion_v**2.0 * score * (
                0.5 if probability_flow else 1.0
            )

            reverse_diffusion_x = torch.zeros_like(diffusion_v)
            reverse_diffusion_v = (
                torch.zeros_like(diffusion_v) if probability_flow else diffusion_v
            )

            return torch.cat((reverse_drift_x, reverse_drift_v), dim=1), torch.cat(
                (reverse_diffusion_x, reverse_diffusion_v), dim=1
            )

        return reverse_sde

    def prior_sampling(self, shape):
        return torch.randn(*shape, device=self.config.device), torch.randn(
            *shape, device=self.config.device
        ) / np.sqrt(self.m_inv)

    def prior_logp(self, u):
        x, v = torch.chunk(u, 2, dim=1)
        N = np.prod(x.shape[1:])

        logx = (
            -N / 2.0 * np.log(2.0 * np.pi)
            - torch.sum(x.view(x.shape[0], -1) ** 2.0, dim=1) / 2.0
        )
        logv = (
            -N / 2.0 * np.log(2.0 * np.pi / self.m_inv)
            - torch.sum(v.view(v.shape[0], -1) ** 2.0, dim=1) * self.m_inv / 2.0
        )
        return logx, logv

    def mean(self, u, t):
        """
        Evaluating the mean of the conditional perturbation kernel.
        """
        x, v = torch.chunk(u, 2, dim=1)

        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)
        coeff_mean = torch.exp(-2.0 * beta_int * self.g)

        mean_x = coeff_mean * (
            2.0 * beta_int * self.g * x + 4.0 * beta_int * self.g**2.0 * v + x
        )
        mean_v = coeff_mean * (-beta_int * x - 2.0 * beta_int * self.g * v + v)
        return torch.cat((mean_x, mean_v), dim=1)

    def var(self, t, var0x=None, var0v=None):
        """
        Evaluating the variance of the conditional perturbation kernel.
        """
        if var0x is None:
            var0x = add_dimensions(
                torch.zeros_like(t, dtype=torch.float64, device=t.device),
                self.config.is_image,
            )
        if var0v is None:
            if self.config.cld_objective == "dsm":
                var0v = torch.zeros_like(t, dtype=torch.float64, device=t.device)
            elif self.config.cld_objective == "hsm":
                var0v = (self.gamma / self.m_inv) * torch.ones_like(
                    t, dtype=torch.float64, device=t.device
                )

            var0v = add_dimensions(var0v, self.config.is_image)

        beta_int = add_dimensions(self.beta_int_fn(t), self.config.is_image)
        multiplier = torch.exp(-4.0 * beta_int * self.g)

        var_xx = (
            var0x
            + (1.0 / multiplier)
            - 1.0
            + 4.0 * beta_int * self.g * (var0x - 1.0)
            + 4.0 * beta_int**2.0 * self.g**2.0 * (var0x - 2.0)
            + 16.0 * self.g**4.0 * beta_int**2.0 * var0v
        )
        var_xv = (
            -var0x * beta_int
            + 4.0 * self.g**2.0 * beta_int * var0v
            - 2.0 * self.g * beta_int**2.0 * (var0x - 2.0)
            - 8.0 * self.g**3.0 * beta_int**2.0 * var0v
        )
        var_vv = (
            self.f**2.0 * ((1.0 / multiplier) - 1.0) / 4.0
            + self.f * beta_int
            - 4.0 * self.g * beta_int * var0v
            + 4.0 * self.g**2.0 * beta_int**2.0 * var0v
            + var0v
            + beta_int**2.0 * (var0x - 2.0)
        )
        return [
            var_xx * multiplier + self.numerical_eps,
            var_xv * multiplier,
            var_vv * multiplier + self.numerical_eps,
        ]

    def mean_and_var(self, u, t, var0x=None, var0v=None):
        return self.mean(u, t), self.var(t, var0x, var0v)

    def noise_multiplier(self, t, var0x=None, var0v=None):
        r"""
        Evaluating the -\ell_t multiplier. Similar to -1/standard deviaton in VPSDE.
        """
        var = self.var(t, var0x, var0v)
        coeff = torch.sqrt(var[0] / (var[0] * var[2] - var[1] ** 2))

        if torch.sum(torch.isnan(coeff)) > 0:
            raise ValueError("Numerical precision error.")

        return -coeff

    def loss_multiplier(self, t):
        """
        Evaluating the "maximum likelihood" multiplier.
        """
        return self.beta_fn(t) * self.f

    def perturb_data(self, batch, t, var0x=None, var0v=None):
        r"""
        Perturbing data according to conditional perturbation kernel with initial variances
        var0x and var0v. Var0x is generally always 0, whereas var0v is 0 for DSM and
        \gamma * M for HSM.
        """
        mean, var = self.mean_and_var(batch, t, var0x, var0v)

        cholesky11 = torch.sqrt(var[0])
        cholesky21 = var[1] / cholesky11
        cholesky22 = torch.sqrt(var[2] - cholesky21**2.0)

        if (
            torch.sum(torch.isnan(cholesky11)) > 0
            or torch.sum(torch.isnan(cholesky21)) > 0
            or torch.sum(torch.isnan(cholesky22)) > 0
        ):
            raise ValueError("Numerical precision error.")

        batch_randn = torch.randn_like(batch, device=batch.device)
        batch_randn_x, batch_randn_v = torch.chunk(batch_randn, 2, dim=1)

        noise_x = cholesky11 * batch_randn_x
        noise_v = cholesky21 * batch_randn_x + cholesky22 * batch_randn_v
        noise = torch.cat((noise_x, noise_v), dim=1)

        perturbed_data = mean + noise
        return perturbed_data, mean, noise, batch_randn

    def get_discrete_step_fn(self, mode, score_fn=None, probability_flow=False):
        if mode == "forward":
            sde_fn = self.sde
        elif mode == "reverse":
            sde_fn = self.get_reverse_sde(
                score_fn=score_fn, probability_flow=probability_flow
            )

        def discrete_step_fn(u, t, dt):
            vec_t = torch.ones(u.shape[0], device=u.device, dtype=torch.float64) * t
            drift, diffusion = sde_fn(u, vec_t)

            drift *= dt
            diffusion *= np.sqrt(dt)

            noise = torch.randn(*u.shape, device=u.device)

            u_mean = u + drift
            u = u_mean + diffusion * noise
            return u, u_mean

        return discrete_step_fn


def add_dimensions(x, is_image):
    if is_image:
        return x[:, None, None, None]
    else:
        return x[:, None]
