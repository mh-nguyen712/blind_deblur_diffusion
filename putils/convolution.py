# %%
import torch
import torch.fft as fft
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
import numpy as np


def bump_function(x, a=1., b=1.):
    """
        Defines a function which is 1 on the interval [-a,a]
        and goes to 0 smoothly on [-a-b,-a]U[a,a+b] using a bump function
        For the discretization of indicator functions, we advise b=1, so that 
        a=0, b=1 yields a Dirac mass.

    Parameters
    ----------
    x : tensor arbitrary size
        input.
    a : float 
        radius (default is 1)
    b : float
        interval on which the function goes to 0. (default is 1)

    Returns
    -------
    v : tensor of the same size as x
        values of psi at x.

    Example 
    -------

    x = np.linspace(-15,15,31)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    Z = bump_function(R,3,1)
    Z = Z / np.sum(Z)

    plt.imshow(Z)
    plt.show()

    """
    v = torch.zeros_like(x)
    v[torch.abs(x) <= a] = 1
    I = (torch.abs(x) > a) * (torch.abs(x) < a + b)
    v[I] = torch.exp(-1. / (1. - ((torch.abs(x[I]) - a) / b)**2)
                     ) / np.exp(-1)
    return v



class Blur2D(object):
    def __init__(self, kernel_size: int = 15,  padding='valid', device=None, dtype=None) -> None:
        self.kernel_size = kernel_size

        self.dirac = torch.zeros(
            (1, 1, kernel_size, kernel_size), device=device, dtype=dtype)
        self.dirac[..., kernel_size//2:kernel_size//2 +
                   1, kernel_size//2:kernel_size//2 + 1] = 1.
        self.padding = padding
        self.device = device
        self.dtype = dtype

    def kernel(*args, **kwargs):
        pass

    def forward(self, x: torch.Tensor, p: float = None) -> torch.Tensor:
        assert x.ndim == 4, f"Input must be 4D, got {x.ndim}D"
        num_channels = x.shape[1]
        if p == 0:
            kernel = self.dirac.expand(num_channels, -1, -1, -1)
        else:
            kernel = self.kernel(p)
            kernel = kernel.expand(num_channels, -1, -1, -1)

        return F.conv2d(x, kernel, padding=self.padding, groups=num_channels)


class GaussianBlur(Blur2D):
    def __init__(self, kernel_size: int = 15, sigma: float = 0.1, *args, **kwargs) -> None:
        super().__init__(kernel_size, *args, **kwargs)

        self.sigma = sigma

        if kernel_size % 2 == 0:
            lin = torch.linspace(-kernel_size//2,
                                 kernel_size//2 - 1, kernel_size, device=self.device, dtype=self.dtype)
        else:
            lin = torch.linspace(-kernel_size//2 + 1,
                                 kernel_size//2, kernel_size, device=self.device, dtype=self.dtype)
        self.grid = lin[None] ** 2 + lin[:, None] ** 2

    def kernel(self, sigma: float = 0.1) -> torch.Tensor:
        if sigma == 0:
            return self.dirac
        else:
            kernel = torch.exp(- self.grid / (2 * sigma ** 2))
            return kernel[None, None] / kernel.sum()


class DefocusBlur(Blur2D):
    def __init__(self, kernel_size: int = 15, radius: float = 0.1, *args, **kwargs) -> None:
        super().__init__(kernel_size, *args, **kwargs)
        self.radius = radius

        if kernel_size % 2 == 0:
            lin = torch.linspace(-kernel_size//2,
                                 kernel_size//2 - 1, kernel_size, device=self.device, dtype=self.dtype)
        else:
            lin = torch.linspace(-kernel_size//2 + 1,
                                 kernel_size//2, kernel_size, device=self.device, dtype=self.dtype)
        self.grid = (lin[None] ** 2 + lin[:, None] ** 2).sqrt()

    def kernel(self, radius: float = 0.1) -> torch.Tensor:
        if radius == 0:
            return self.dirac
        else:
            kernel = bump_function(self.grid, a=radius).type(self.grid.dtype)
            kernel = gaussian_blur(kernel[None, None], sigma=0.5, kernel_size=(5, 5))
            return kernel / kernel.sum()


def wspace(t):
    '''
    This function constructs a linearly-spaced vector of angular
    frequencies that correspond to the points in an FFT spectrum.
    The second half of the vector is aliased to negative
    frequencies.
    '''

    nt = t.size(0)
    dt = t[1] - t[0]
    t = t[-1] - t[0] + dt

    w = 2 * torch.pi * torch.arange(nt) / t
    kv = torch.where(w >= torch.pi / dt)
    w[kv] -= 2 * torch.pi / dt
    return w


class AiryDisc(Blur2D):
    def __init__(self, kernel_size: int = 15, radius: float = 0.1, *args, **kwargs) -> None:
        super().__init__(kernel_size, *args, **kwargs)
        self.radius = radius

        w = fft.fftfreq(kernel_size).to(device=self.device, dtype=self.dtype)
        wx, wy = torch.meshgrid(fft.fftshift(w), fft.fftshift(w))
        self.grid = (wx ** 2 + wy ** 2).sqrt()

    def kernel(self, radius: float = 0.1) -> torch.Tensor:

        if radius == 0:
            return self.dirac
        else:
            radius = 3.8317 * 2 * np.pi ** 2 / radius
            kernel = 2 * torch.pi * radius * \
                torch.special.bessel_j1(self.grid * radius) / self.grid
            kernel[self.grid == 0] = torch.pi * radius ** 2
            kernel = kernel.abs() ** 2
            return kernel[None, None] / kernel.sum()


class MotionBlur(Blur2D):
    def __init__(self, kernel_size: int = 15, length: int = 1, angle: float = 0., *args, **kwargs) -> None:
        super().__init__(kernel_size, *args, **kwargs)
        self.length = length
        self.angle = angle

        self.base_kernel = torch.zeros(
            self.kernel_size, self.kernel_size, device=self.device, dtype=self.dtype)

        if kernel_size % 2 == 0:
            self.lin = torch.linspace(-kernel_size//2,
                                      kernel_size//2 - 1, kernel_size, device=self.device, dtype=self.dtype)
        else:
            self.lin = torch.linspace(-kernel_size//2 + 1,
                                      kernel_size//2, kernel_size, device=self.device, dtype=self.dtype)

    def kernel(self, length: int = 1, angle=None):
        if angle is None:
            angle = self.angle
        if length == 0:
            return self.dirac
        else:
            kernel = self.base_kernel.clone()
            center = int(self.kernel_size//2)

            if angle == 0:
                kernel[center:center + 1,
                       :] = bump_function(self.lin, a=length)
            elif angle == 90:
                kernel[:, center] = bump_function(self.lin, a=length)
            elif angle == 45:
                kernel[list(reversed(range(self.kernel_size))), list(range(
                    self.kernel_size))] = bump_function(self.lin, a=length)

            kernel = gaussian_blur(kernel[None, None], sigma=0.5, kernel_size=(5, 5))
            return kernel / kernel.sum()


def get_blur_class(name, *args, **kwargs):
    if name.lower() == 'gaussian':
        return GaussianBlur(*args, **kwargs)
    elif name.lower() == 'defocus':
        return DefocusBlur(*args, **kwargs)
    elif name.lower() == 'airy':
        return AiryDisc(*args, **kwargs)
    elif name.lower() == 'motion':
        return MotionBlur(*args, **kwargs)
    else:
        raise NotImplementedError(f'The {name} blur is not implemented')


def conv2d(input, kernel, padding='same', stride=1):
    '''
    A helper function to perform the convolution between each image 
    in the batch and each PSF
    Args:
        input [Tensor] (B, C, H, W) batch of images
        kernel [Tensor] (B, 1, H_k, W_k) batch of PSFs
    Returns: output [Tensor] of shape(B, C, H', W')
        where output[i] = input[i] * kernel[i]
    '''
    assert input.dim() == kernel.dim() == 4, 'Input and PSF must be 4D tensors'
    assert kernel.size(
        1) == 1, 'Only supported for single kernel per image (for all channels)'

    if kernel.size(0) == 1:
        kernel = kernel.expand(input.size(0), -1, -1, -1)
        # print('Warning: Input and PSF must have the same batch size, the PSF is expanded!')

    # Move batch dim into channels
    B, C, H, W = input.size()
    input = input.view(1, -1, H, W)

    h, w = kernel.size(-2), kernel.size(-1)
    kernel = kernel.expand(-1, C, -1, -1).reshape(B * C, -1, h, w)

    output = F.conv2d(input, kernel, padding=padding,
                      stride=stride, groups=B * C)

    return output.view(B, C, output.size(-2), -1)


def conv2d_transpose(input, kernel, padding=None, stride=1):
    '''
    A helper function to perform the convolution transpose between each image 
    in the batch and each PSF
    Args:
        input [Tensor] (B, C, H, W) batch of images
        kernel [Tensor] (B, 1, H_k, W_k) batch of PSFs
    Returns: output [Tensor] of shape(B, C, H', W')
        where output[i] = input[i] * kernel[i]

    Padding conditions:
        'valid' <=> 0
        'same' <=> kernel_size // 2
    '''
    assert input.dim() == kernel.dim() == 4, 'Input and PSF must be 4D tensors'
    assert kernel.size(
        1) == 1, 'Only supported for single kernel per image (for all channels)'
    if kernel.size(0) == 1:
        kernel = kernel.expand(input.size(0), -1, -1, -1)
        # print('Warning: Input and PSF must have the same batch size, the PSF is expanded!')

    if padding is None:
        padding = (kernel.size(-2) // 2, kernel.size(-1) // 2)
    # Move batch dim into channels
    B, C, H, W = input.size()
    input = input.view(1, -1, H, W)

    h, w = kernel.size(-2), kernel.size(-1)
    kernel = kernel.expand(-1, C, -1, -1).reshape(B * C, -1, h, w)

    output = F.conv_transpose2d(input, kernel, padding=padding,
                                stride=stride, groups=B * C)

    return output.view(B, C, output.size(-2), -1)
