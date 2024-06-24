import torch
from torch.nn import functional as F

def unit_Weiner_process(x, noise_source=torch.randn_like):
    """
    Computes dynamic update of a unit Weiner process given the input tensor
    """
    return noise_source(x)

def unit_OU_process(x, noise_source=torch.randn_like):
    """
    Computes dynamic update of a unit Ornstein-Uhlenbeck process given the input tensor.

    Args:
        x: The input tensor.
        noise_source: A function that generates noise with the same shape as x. 
                      Default is torch.randn_like.

    Returns:
        The result of applying the unit UH process to x.
    """
    return -x + noise_source(x)

def anticorrelated_noise(x, noise_source=torch.randn_like):
    """
    Generates anti-correlated noise in the shape of the input tensor.

    Args:
        x (torch.Tensor): The input tensor.
        noise_source (callable): A function that generates noise tensor with the same shape as x. 
                                Default is torch.randn_like.

    Returns:
        torch.Tensor: Anticorrelated noise in the shape of x.
    """
    a = noise_source(x)
    diff = a[1:] - a[:-1]
    return F.pad(diff, (0, 0) * (x.dim() - 1) + (1, 0))

def correlated_noise(x, noise_source=torch.randn_like):
    """
    Generates correlated noise in the shape of the input tensor.

    Args:
        x (torch.Tensor): The input tensor.
        noise_source (callable): A function that generates noise tensor with the same shape as x. 
                                Default is torch.randn_like.

    Returns:
        torch.Tensor: Correlated noise in the shape of x.

    """
    a = noise_source(x)
    diff = a[1:] + a[:-1]
    return F.pad(diff, (0, 0) * (x.dim() - 1) + (1, 0))