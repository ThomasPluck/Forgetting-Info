import torch
from torch.nn import functional as F
import functools
import einops

"""
In this script we define "kernel operators" on kernel gram matrices. Each kernel operator needs:

1. Two inputs x and y tensors with equal length dimensions on some axes.
2. An "axis" argument which tells the model which dimensions are considered
    elements to compute Gram matrices on. 
"""

def _off_diagonal(x : torch.Tensor, offset=1, axis=None):
    """
    Compute the off-diagonal elements of a matrix, in general, we assume that the first
    axis is the "sample axis" and so cant be a member of axis.

    Args:
        x (torch.Tensor): The input tensor.
        offset (int, optional): The offset for the off-diagonal elements. Defaults to 1.
        axis (tuple, optional): The axis along which to compute the off-diagonal elements. 
            If None, an error will be raised. Defaults to None.

    Returns:
        torch.Tensor: The off-diagonal elements of the matrix.

    Raises:
        ValueError: If axis is None or if the length of axis is not equal to len(x) - 1.

    """
    if 0 in axis:
        raise ValueError("0 is the sample axis and can not be in the axis tuple.")
    if offset < 1:
        raise ValueError(f"offset = {offset} must be greater than 0!")
    
    # Create slices for samples_1 and samples_2
    samples_1 = x[offset:]
    samples_2 = x[:-offset]

    return samples_1, samples_2

def gaussian_kernel_gram_off_diagonal(x: torch.Tensor, offset=1, axis=None, sigma=1):
    """
    Compute the off-diagonal elements of the Gaussian kernel Gram matrix.

    Args:
        x (torch.Tensor): Input tensor.
        offset (int, optional): Offset from the diagonal. Defaults to 1.
        axis (tuple or None, optional): Axes along which to compute off-diagonal elements. Defaults to None.
        sigma (float, optional): Standard deviation of the Gaussian kernel. Defaults to 1.

    Returns:
        torch.Tensor: Off-diagonal elements of the Gaussian kernel Gram matrix.
    """
    samples_1, samples_2 = _off_diagonal(x, offset, axis)
    axis = (0,) if axis is None else axis
    return torch.exp(-torch.sum((samples_1-samples_2)**2, dim=axis) / (2 * sigma**2))

def cosine_kernel_gram_off_diagonal(x: torch.Tensor, offset=1, axis=None):
    """
    Compute the off-diagonal elements of the Cosine kernel Gram matrix.

    Args:
        x (torch.Tensor): Input tensor.
        offset (int, optional): Offset from the diagonal. Defaults to 1.
        axis (tuple or None, optional): Axes along which to compute off-diagonal elements. Defaults to None.

    Returns:
        torch.Tensor: Off-diagonal elements of the Cosine kernel Gram matrix.
    """
    samples_1, samples_2 = _off_diagonal(x, offset, axis)
    axis = (0,) if axis is None else axis
    samples_1 = F.normalize(samples_1, dim=axis)
    samples_2 = F.normalize(samples_2, dim=axis)
    return torch.sum(samples_1 * samples_2, dim=axis)

def dot_kernel_gram_off_diagonal(x: torch.Tensor, offset=1, axis=None):
    """
    Compute the off-diagonal elements of the Dot product kernel Gram matrix.

    Args:
        x (torch.Tensor): Input tensor.
        offset (int, optional): Offset from the diagonal. Defaults to 1.
        axis (tuple or None, optional): Axes along which to compute off-diagonal elements. Defaults to None.

    Returns:
        torch.Tensor: Off-diagonal elements of the Dot product kernel Gram matrix.
    """
    samples_1, samples_2 = _off_diagonal(x, offset, axis)
    axis = (0,) if axis is None else axis
    return torch.sum(samples_1 * samples_2, dim=axis)

def l2_kernel_gram_off_diagonal(x: torch.Tensor, offset=1, axis=None):
    """
    Compute the off-diagonal elements of the L2 distance kernel Gram matrix.

    Args:
        x (torch.Tensor): Input tensor.
        offset (int, optional): Offset from the diagonal. Defaults to 1.
        axis (tuple or None, optional): Axes along which to compute off-diagonal elements. Defaults to None.

    Returns:
        torch.Tensor: Off-diagonal elements of the L2 distance kernel Gram matrix.
    """
    samples_1, samples_2 = _off_diagonal(x, offset, axis)
    axis = (0,) if axis is None else axis
    return torch.sqrt(torch.sum((samples_1 - samples_2)**2, dim=axis))

def l1_kernel_gram_off_diagonal(x: torch.Tensor, offset=1, axis=None):
    """
    Compute the off-diagonal elements of the L1 distance kernel Gram matrix.

    Args:
        x (torch.Tensor): Input tensor.
        offset (int, optional): Offset from the diagonal. Defaults to 1.
        axis (tuple or None, optional): Axes along which to compute off-diagonal elements. Defaults to None.

    Returns:
        torch.Tensor: Off-diagonal elements of the L1 distance kernel Gram matrix.
    """
    samples_1, samples_2 = _off_diagonal(x, offset, axis)
    axis = (0,) if axis is None else axis
    return torch.sum(torch.abs(samples_1 - samples_2), dim=axis)

def lp_kernel_gram_off_diagonal(x: torch.Tensor, offset=1, axis=None, p=2):
    """
    Compute the off-diagonal elements of the Lp distance kernel Gram matrix.

    Args:
        x (torch.Tensor): Input tensor.
        offset (int, optional): Offset from the diagonal. Defaults to 1.
        axis (tuple or None, optional): Axes along which to compute off-diagonal elements. Defaults to None.
        p (float, optional): The order of the norm. Defaults to 2.

    Returns:
        torch.Tensor: Off-diagonal elements of the Lp distance kernel Gram matrix.
    """
    samples_1, samples_2 = _off_diagonal(x, offset, axis)
    axis = (0,) if axis is None else axis
    return (torch.sum((samples_1 - samples_2)**p, dim=axis))**(1/p)