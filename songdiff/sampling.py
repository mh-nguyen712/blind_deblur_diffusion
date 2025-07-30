"""
Sampling methods
Modified from: https://github.com/yang-song/score_sde_pytorch/
"""

import functools

import torch
from torch import Tensor
import numpy as np
import abc

from .sde_utils import from_flattened_numpy, to_flattened_numpy
from scipy import integrate
from .sde_lib import SDE, VESDE, VPSDE, subVPSDE

from typing import Callable, Tuple, List, Union, Optional
from ml_collections import ConfigDict

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """
    A decorator for registering predictor classes.
    """
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls

        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """
    A decorator for registering corrector classes.
    """

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.
    Args:
            model: The score model.
            train: `True` for training and `False` for evaluation.

    Returns:
            A model function.
    """

    def model_fn(x: Tensor, labels: Union[Tensor, float]):
        """Compute the output of the score-based model.

        Args:
        x: A mini-batch of input data.
        labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
                for different models.

        Returns:
        A tuple of (model output, new mutable states)
        """
        if not train:
            with torch.no_grad():
                model.eval()
                return model(x, labels)
        else:
            model.train()
            return model(x, labels)

    return model_fn


def get_score_fn(sde: SDE, model: Callable, train: bool = False, continuous: bool = False):
    """
    Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
            sde: An `SDE` object that represents the forward SDE.
            model: A score model.
            train: `True` for training and `False` for evaluation.
            continuous: If `True`, the score-based model is expected to directly take continuous time steps.

    Returns:
            A score function.
    """
    model_fn = get_model_fn(model, train=train)

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
        def score_fn(x, t):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = model_fn(x, labels)
                std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.n_time_steps - 1)
                score = model_fn(x, labels)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[
                    labels.long()]

            score = -score / std[:, None, None, None]
            return score

    elif isinstance(sde, VESDE):
        def score_fn(x, t):
            if continuous:
                labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.n_time_steps - 1
                labels = torch.round(labels).long()

            score = model_fn(x, labels)
            return score

    else:
        raise NotImplementedError(
            f"SDE class {sde.__class__.__name__} not yet supported.")

    return score_fn


def get_sampling_fn(config: ConfigDict, sde: SDE, shape: Union[List, Tuple], inverse_scaler: Callable, eps: float):
    """
    Create a sampling function (on the reverse SDE).

    Args:
            config: A `ml_collections.ConfigDict` object that contains all configuration information.
            sde: A `SDE` object that represents the forward SDE.
            shape: A sequence of integers representing the expected shape of a single sample.
            inverse_scaler: The inverse data normalizer function.
            eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
            A function that takes random states and a replicated training state and outputs samples with the
            trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == 'ode':
        sampling_fn = get_ode_sampler(sde=sde,
                                      shape=shape,
                                      inverse_scaler=inverse_scaler,
                                      denoise=config.sampling.noise_removal,
                                      eps=eps,
                                      device=config.device)

    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(sde=sde,
                                     shape=shape,
                                     predictor=predictor,
                                     corrector=corrector,
                                     inverse_scaler=inverse_scaler,
                                     snr=config.sampling.snr,
                                     n_steps=config.sampling.n_steps_each,
                                     probability_flow=config.sampling.probability_flow,
                                     continuous=config.training.continuous,
                                     denoise=config.sampling.noise_removal,
                                     eps=eps,
                                     device=config.device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """
    The abstract class for a predictor algorithm.
    """

    def __init__(self, sde: SDE, score_fn: Callable, probability_flow: bool = False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x: Tensor, t: Union[Tensor, float]):
        """
        One update of the predictor. 

        Args:
        x: A PyTorch tensor representing the current state
        t: a float (or 0 or 1 dim tensor) representing the current time step.

        Returns:
        x: A PyTorch tensor of the next state.
        x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """
    The abstract class for a corrector algorithm.
    """

    def __init__(self, sde: SDE, score_fn: Callable, snr, n_steps: int):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x: Tensor, t: Union[Tensor, float]):
        """
        One update of the corrector.

        Args:
        x: A PyTorch tensor representing the current state
        t: a float (or 0 or 1 dim tensor) representing the current time step.

        Returns:
        x: A PyTorch tensor of the next state.
        x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    """
    Euler Maruyama discretization for the reverse SDE
    """

    def __init__(self, sde: SDE, score_fn: Callable, probability_flow: bool = False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x: Tensor, t: Union[Tensor, float]):
        dt = -self.sde.T / self.rsde.n_time_steps
        z = torch.randn_like(x)

        # Get drift and diffusion coefficient of the reverse SDE
        drift, diffusion = self.rsde.sde(x, t)
        # Perform Euler Maruyama discretization
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    """
    Sampling by solving the reverse SDE using discretization (default to Euler Maruyama)
    Which composes:
            -> Discretize forward SDE: discretized f and g 
            -> Compute discretized version of drift and diffusion of reverse SDE
            -> Perform Euler-Maruyama rule
    For VP SDE: DDPM discretization is used
    For VE SDE: SMLD (NCSN) discretization is used
    """

    def __init__(self, sde: SDE, score_fn: Callable, probability_flow: bool = False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x: Tensor, t: Union[Tensor, float]):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        # x + drift * d t_bar = x - drift * dt
        x_mean = x - f
        # add g(t)sqrt(dt) * z
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
    """
    The ancestral sampling predictor (or only predictor sampling). Currently only supports VE/VP SDEs.
    """

    def __init__(self, sde: SDE, score_fn: Callable, probability_flow: bool = False):
        super().__init__(sde, score_fn, probability_flow)

        if not isinstance(sde, VPSDE) and not isinstance(sde, VESDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported.")

        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, x: Tensor, t: Union[Tensor, float]):
        """
        One update of the predictor of the VE SDE.

        Args:
        x: A PyTorch tensor representing the current state
        t: a float (or 0 or 1 dim tensor) representing the current time step.

        Returns:
        x: A PyTorch tensor of the next state.
        x_mean: A PyTorch tensor. The next state without random noise.
        """
        sde = self.sde
        timestep = int(t * (sde.n_time_steps - 1) / sde.T)
        # The discrete sigma at current time step
        sigma = sde.discrete_sigmas[timestep]
        # Eq (47) in the original paper
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(
            t), sde.discrete_sigmas[timestep - 1].to(t.device))
        score = self.score_fn(x, t)
        x_mean = x + score * \
            (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
        std = torch.sqrt(
            (adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x: Tensor, t: Union[Tensor, float]):
        """
        One update of the predictor of the VE SDE.
        Using DDPM ancestral sampling

        Args:
        x: A PyTorch tensor representing the current state
        t: a float (or 0 or 1 dim tensor) representing the current time step.

        Returns:
        x: A PyTorch tensor of the next state.
        x_mean: A PyTorch tensor. The next state without random noise.
        """

        sde = self.sde
        timestep = int(t * (sde.n_time_steps - 1) / sde.T)
        # The discrete beta at current time step
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t)
        # DDPM sampling, as in Alg. 2 of DDPM paper or Eq (4) of SDE paper, with interpretation of score model
        x_mean = (x + beta[:, None, None, None] * score) / \
            torch.sqrt(1. - beta)[:, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update_fn(self, x: Tensor, t: Union[Tensor, float]):
        if isinstance(self.sde, VESDE):
            return self.vesde_update_fn(x, t)
        elif isinstance(self.sde, VPSDE):
            return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde: SDE, score_fn: Callable, probability_flow: bool = False):
        pass

    def update_fn(self, x: Tensor, t: Union[Tensor, float]):
        return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    """
    Using MCMC Langevin dynamics as corrector by solving overdamped Langevin Itô diffusions 
    """

    def __init__(self, sde: SDE, score_fn: Callable, snr: float, n_steps: int = 1):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, VPSDE) \
                and not isinstance(sde, VESDE) \
                and not isinstance(sde, subVPSDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x: Tensor, t: Union[Tensor, float]):
        """
        Performs n_steps of Langevin sampling
        """
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr

        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            timestep = (t * (sde.n_time_steps - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            # score: grad_x log p_t (x)
            grad = score_fn(x, t)
            # random noise z
            noise = torch.randn_like(x)
            # compute the step size, following the original paper of SDE modeling
            grad_norm = torch.norm(grad.reshape(
                grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(
                noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            # perform Langevin dynamics step
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
    """
    The original annealed Langevin dynamics predictor in NCSN/NCSNv2.
    """

    def __init__(self, sde: SDE, score_fn: Callable, snr: float, n_steps: int = 1):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, VPSDE) \
                and not isinstance(sde, VESDE) \
                and not isinstance(sde, subVPSDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x: Tensor, t: Union[Tensor, float]):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr

        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            timestep = int(t * (sde.n_time_steps - 1) / sde.T)
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = 2 * alpha * (target_snr * std) ** 2
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x: Tensor, t: float):
        return x, x


def shared_predictor_update_fn(x: Tensor, t: Union[Tensor, float], sde: SDE, model: Callable,
                               predictor: Optional[Predictor] = None, probability_flow: bool = False, continuous: bool = False):
    """
    A wrapper that configures and returns the update function of predictors.
    """
    score_fn = get_score_fn(sde, model, train=False, continuous=continuous)

    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x: Tensor, t: Union[Tensor, float], sde: SDE, model: Callable,
                               corrector: Optional[Corrector] = None, continuous: bool = False, snr: float = 1., n_steps: int = 1000):
    """
    A wrapper that configures and returns the update function of correctors.
    """
    score_fn = get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t)


def get_pc_sampler(sde: SDE, shape: Union[List, Tuple], predictor: Predictor, corrector: Predictor,
                   inverse_scaler: Callable, snr: float,
                   n_steps: int = 1, probability_flow: bool = False, continuous: bool = False,
                   denoise: bool = True, eps: float = 1e-3, device='cuda'):
    """
    Create a Predictor-Corrector (PC) sampler.

    Args:
            sde: An `SDE` object representing the forward SDE.
            shape: A sequence of integers. The expected shape of a single sample.
            predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
            corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
            inverse_scaler: The inverse data normalizer.
            snr: A `float` number. The signal-to-noise ratio for configuring correctors.
            n_steps: An integer. The number of corrector steps per predictor update.
            probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
            continuous: `True` indicates that the score model was continuously trained.
            denoise: If `True`, add one-step denoising to the final samples.
            eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` instead of 0 to avoid numerical issues.
            device: PyTorch device.

    Returns:
            A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    # Alg. 1 in the original SDE modeling paper
    def pc_sampler(model):
        """ The PC sampler function.

        Args:
        model: A score model.
        Returns:
        Samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(
                sde.T, eps, sde.n_time_steps, device=device)

            for i in range(sde.n_time_steps):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                # print(t.device)
                # Why corrector before predictor?
                x, x_mean = corrector_update_fn(x, vec_t, model=model)
                x, x_mean = predictor_update_fn(x, vec_t, model=model)

            return inverse_scaler(x_mean if denoise else x), sde.n_time_steps * (n_steps + 1)

    return pc_sampler


def get_ode_sampler(sde: SDE, shape: Union[List, Tuple], inverse_scaler: Callable,
                    denoise: bool = False, rtol: float = 1e-5, atol: float = 1e-5,
                    method: str = 'RK45', eps: float = 1e-3, device='cuda'):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
            sde: An `SDE` object that represents the forward SDE.
            shape: A sequence of integers. The expected shape of a single sample.
            inverse_scaler: The inverse data normalizer.
            denoise: If `True`, add one-step denoising to final samples.
            rtol: A `float` number. The relative tolerance level of the ODE solver.
            atol: A `float` number. The absolute tolerance level of the ODE solver.
            method: A `str`. The algorithm used for the black-box ODE solver.
            See the documentation of `scipy.integrate.solve_ivp`.
            eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
            device: PyTorch device.

    Returns:
            A sampling function that returns samples and the number of function evaluations during sampling.
    """

    def denoise_update_fn(model: Callable, x: Tensor):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(
            sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps)
        return x

    def drift_fn(model: Callable, x: Tensor, t: float):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(model: Callable, z: Tensor = None):
        """
        The probability flow ODE sampler with black-box ODE solver.

        Args:
        model: A score model.
        z: If present, generate samples from latent code `z`.
        Returns:
        samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(
                    device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(
                solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x)

            x = inverse_scaler(x)
            return x, nfe
    return ode_sampler
