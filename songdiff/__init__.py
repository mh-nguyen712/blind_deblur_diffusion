from .prior import DIFFUSION_POTENTIAL, DIFFUSION_GRADIENT

def get_potential_fn(name, *args, **kwargs):
    if name in DIFFUSION_POTENTIAL.keys():
        return DIFFUSION_POTENTIAL[name](*args)
    else:
        raise ValueError(f"No prior registered with name: {name}")


def get_gradient_fn(name, *args, **kwargs):
    return DIFFUSION_GRADIENT[name](*args, **kwargs)