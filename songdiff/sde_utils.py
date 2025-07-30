"""
Some useful utilities for working with PyTorch tensor and Numpy array

"""
import os
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from einops import rearrange, asnumpy
import pathlib
from PIL import Image
from typing import BinaryIO, List, Optional, Tuple, Union

TensorArray = Union[Tensor, np.ndarray]


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def is_one_dim_tensor(x):
    """
    Check if input is an one-dimensional tensor or not
    """
    if isinstance(x, Tensor):
        return True if x.ndim == 1 else False
    else:
        return False


def to_numpy(input: Tensor):
    return input.detach().cpu().numpy()


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        print(
            f"WARNING: No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        print(loaded_state['model'].keys())
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state
