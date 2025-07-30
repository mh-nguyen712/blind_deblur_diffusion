import torch
import os
from .likelihood import get_likelihood_fn
from .sde_lib import VESDE
from .configs.ve import (
    celebahq_256_ncsnpp_continuous,
    ffhq_256_ncsnpp_continuous,
    ffhq_ncsnpp_continuous,
    church_ncsnpp_continuous,
)
from .models import utils as mutils
from .sampling import get_score_fn
import gdown 
from .models.ema import ExponentialMovingAverage

DIFFUSION_POTENTIAL = {}  # negative log-likelihood
DIFFUSION_GRADIENT = {}

default_ncsnpp = dict(
    atol=1e-4,
    rtol=1e-4,
    backends="scipy",
)

def register_prior(func=None, *, name=None):
    """
    A decorator for registering prior functions.
    """

    def _register(func):
        if name is None:
            local_name = func.__name__
        else:
            local_name = name
        if local_name in DIFFUSION_POTENTIAL.keys():
            raise ValueError(
                f"Already registered prior with name: {local_name}")
        DIFFUSION_POTENTIAL[local_name] = func

        return func

    if func is None:
        return _register
    else:
        return _register(func)


def register_gradient(func=None, *, name=None):
    """
    A decorator for registering prior functions.
    """

    def _register(func):
        if name is None:
            local_name = func.__name__
        else:
            local_name = name
        if local_name in DIFFUSION_GRADIENT.keys():
            raise ValueError(
                f"Already registered prior gradient with name: {local_name}"
            )
        DIFFUSION_GRADIENT[local_name] = func

        return func

    if func is None:
        return _register
    else:
        return _register(func)


# Register all score functions
def get_score_model(name, **factory_kwargs):
    root_path = os.path.expanduser("~")

    if name.lower() == "ncsnpp_celeba":
        config = celebahq_256_ncsnpp_continuous.get_config()
        ckpt_url = 'https://drive.google.com/uc?export=download&id=1ocvHVzAeYtwIRFPgqG1CPPHdXUzY85UG'
    elif name.lower() == "ncsnpp_ffhq256":
        config = ffhq_256_ncsnpp_continuous.get_config()
        ckpt_url = 'https://drive.google.com/uc?export=download&id=1-mtdSwuefIZA0n85QWScQo2WRvJNWwUy'
    elif name.lower() == "ncsnpp_ffhq1024":
        config = ffhq_ncsnpp_continuous.get_config()
        ckpt_url = 'https://drive.google.com/file/d/1sXrlgTC6U2jzWCIUZTbRcY3AbOJfcbbu/view?usp=drive_link'
    elif name.lower() == "ncsnpp_church":
        config = church_ncsnpp_continuous.get_config()
        ckpt_url = 'https://drive.google.com/file/d/1KUZbguxtrqlXtQINwa9OCdOIu1tRLASm/view?usp=drive_link'

    sde = VESDE(
        sigma_min=config.model.sigma_min,
        sigma_max=config.model.sigma_max,
        n_time_steps=config.model.num_scales,
    )
    ckpt_path = os.path.join(root_path, "ckpt", f"{name}.pth")
    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        gdown.download(ckpt_url, ckpt_path, quiet=False)
        
    score_model = mutils.create_model(config)
    
    # optimizer = mutils.get_optimizer(config, score_model.parameters())
    # ema = ExponentialMovingAverage(score_model.parameters(),
    #                             decay=config.model.ema_rate)
    # state = dict(step=0, optimizer=optimizer,
    #             model=score_model, ema=ema)

    # state = mutils.restore_checkpoint(ckpt_path, state, config.device)
    # ema.copy_to(score_model.parameters())
    
    pretrained_state_dict = torch.load(ckpt_path)['model']
    pretrained_state_dict_corrected = {}
    for key, value in pretrained_state_dict.items():
        new_key = ".".join(key.split(".")[1:])
        pretrained_state_dict_corrected[new_key] = value.to(**factory_kwargs)

    score_model.load_state_dict(pretrained_state_dict_corrected, strict=False)
    
    score_model = score_model.to(**factory_kwargs)
    is_continuous = config.training.continuous
    return score_model, sde, is_continuous, config 


def wrapped_score_model_time(score_model, x, t):
    factory_kwargs = {"device": x.device, "dtype": x.dtype}
    if isinstance(t, float):
        t = torch.tensor([t] * x.size(0), **factory_kwargs)
    elif isinstance(t, torch.Tensor):
        t = t.to(**factory_kwargs)
        if t.size(0) == 1 and t.size(0) != x.size(0):
            t = t.expand(x.size(0))
        else:
            assert t.size(0) == x.size(
                0), "Time and image must have the same batch size"
    return score_model(x, t)


@register_gradient(name="ncsnpp_celeba")
def ncsnpp_celeba_score_fn(**factory_kwargs):
    score_model, sde, is_continuous, _ = get_score_model(
        "ncsnpp_celeba", **factory_kwargs)
    score_fn = get_score_fn(sde, score_model, False, is_continuous)
    return lambda x, t: wrapped_score_model_time(score_fn, x, t)


@register_gradient(name="ncsnpp_ffhq256")
def ncsnpp_ffhq256_score_fn(**factory_kwargs):
    score_model, sde, is_continuous, _ = get_score_model(
        "ncsnpp_ffhq256", **factory_kwargs
    )
    score_fn = get_score_fn(sde, score_model, False, is_continuous)
    return lambda x, t: wrapped_score_model_time(score_fn, x, t)


@register_gradient(name="ncsnpp_ffhq1024")
def ncsnpp_ffhq1024_score_fn(**factory_kwargs):
    score_model, sde, is_continuous, _ = get_score_model(
        "ncsnpp_ffhq1024", **factory_kwargs
    )
    score_fn = get_score_fn(sde, score_model, False, is_continuous)
    return lambda x, t: wrapped_score_model_time(score_fn, x, t)


@register_gradient(name="ncsnpp_church")
def ncsnpp_church_score_fn(**factory_kwargs):
    score_model, sde, is_continuous, _ = get_score_model(
        "ncsnpp_church", **factory_kwargs)
    score_fn = get_score_fn(sde, score_model, False, is_continuous)
    return lambda x, t: wrapped_score_model_time(score_fn, x, t)


# Register all likelihood functions
@register_prior(name="ncsnpp_celeba")
def ncsnpp_celeba_potential_fn(options=default_ncsnpp, **factory_kwargs):
    score_model, sde, _, _ = get_score_model("ncsnpp_celeba", **factory_kwargs)
    likelihood_fn = get_likelihood_fn(sde, **options)

    return lambda x: likelihood_fn(score_model, x)


@register_prior(name="ncsnpp_ffhq256")
def ncsnpp_ffhq256_potential_fn(options=default_ncsnpp, **factory_kwargs):
    score_model, sde, _, _ = get_score_model("ncsnpp_ffhq256", **factory_kwargs)
    likelihood_fn = get_likelihood_fn(sde, **options)

    return lambda x: likelihood_fn(score_model, x)


@register_prior(name="ncsnpp_ffhq1024")
def ncsnpp_ffhq1024_potential_fn(options=default_ncsnpp, **factory_kwargs):
    score_model, sde, _, _ = get_score_model("ncsnpp_ffhq1024", **factory_kwargs)
    likelihood_fn = get_likelihood_fn(sde, **options)

    return lambda x: likelihood_fn(score_model, x)


@register_prior(name="ncsnpp_church")
def ncsnpp_church_potential_fn(options=default_ncsnpp, **factory_kwargs):
    score_model, sde, _, _ = get_score_model("ncsnpp_church", **factory_kwargs)
    likelihood_fn = get_likelihood_fn(sde, **options)

    return lambda x: likelihood_fn(score_model, x)
