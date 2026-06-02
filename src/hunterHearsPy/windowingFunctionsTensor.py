"""Create PyTorch tensor windowing functions."""
from __future__ import annotations

from collections.abc import Callable
from hunterHearsPy import cosineWings, equalPower, halfsine, tukey, WindowingFunction
from torch.types import Device
from typing import Any, TypeVar
import torch

callableReturnsNDArray = TypeVar('callableReturnsNDArray', bound=Callable[..., WindowingFunction])

def _convertToTensor(*arguments: Any, callableTarget: callableReturnsNDArray, device: Device, **keywordArguments: Any) -> torch.Tensor:
    arrayTarget = callableTarget(*arguments, **keywordArguments)
    return torch.tensor(data=arrayTarget, dtype=torch.float32, device=device)

def cosineWingsTensor(lengthWindow: int, ratioTaper: float | None=None, device: Device | None=None) -> torch.Tensor:
    """Generate a cosine-tapered windowing function with a flat center and tapered ends.

    This function creates a NumPy array [1] of coefficients that rise smoothly from 0 to 1, stay
    flat in the middle, and mirror the taper at the end. It uses `numpy.linspace` [2] to sample the
    taper and `numpy.cos` [3] to shape it. It acts on `lengthWindow` and `ratioTaper` and returns
    the windowing coefficients.

    Parameters
    ----------
    lengthWindow : int
        Total length of the windowing function.
    ratioTaper : float | None = None
        Ratio of taper length to windowing-function length. The value must be between 0 and 1,
        inclusive.
    device : Device = torch.device(device='cpu')
        PyTorch device for tensor allocation.

    Returns
    -------
    windowingFunction : WindowingFunction
        NumPy array of windowing coefficients with cosine tapers.

    See Also
    --------
    `hunterHearsPy.optionalPyTorch.cosineWingsTensor`
        Tensor version of this windowing function.

    References
    ----------
    [1] NumPy `ndarray`.
        https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
    [2] NumPy `linspace`.
        https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
    [3] NumPy `cos`.
        https://numpy.org/doc/stable/reference/generated/numpy.cos.html

    """
    device = device or torch.device(device='cpu')
    return _convertToTensor(lengthWindow, ratioTaper, callableTarget=cosineWings, device=device)

def equalPowerTensor(lengthWindow: int, ratioTaper: float | None=None, device: Device | None=None) -> torch.Tensor:
    """Generate an equal-power taper for crossfades.

    This function creates a NumPy array [1] of coefficients that follow a square-root taper at both
    ends and a flat center. It uses `numpy.linspace` [2] to sample the taper and `numpy.sqrt` [3] to
    convert it to equal-power scaling. It acts on `lengthWindow` and `ratioTaper` and returns the
    windowing coefficients.

    Parameters
    ----------
    lengthWindow : int
        Total length of the windowing function.
    ratioTaper : float | None = None
        Ratio of taper length to windowing-function length. The value must be between 0 and 1,
        inclusive.
    device : Device = torch.device(device='cpu')
        PyTorch device for tensor allocation.

    Returns
    -------
    windowingFunction : WindowingFunction
        NumPy array of windowing coefficients with tapers.

    See Also
    --------
    `hunterHearsPy.optionalPyTorch.equalPowerTensor`
        Tensor version of this windowing function.

    References
    ----------
    [1] NumPy `ndarray`.
        https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
    [2] NumPy `linspace`.
        https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
    [3] NumPy `sqrt`.
        https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html

    """
    device = device or torch.device(device='cpu')
    return _convertToTensor(lengthWindow, ratioTaper, callableTarget=equalPower, device=device)

def halfsineTensor(lengthWindow: int, device: Device | None=None) -> torch.Tensor:
    """Generate a half-sine windowing function.

    This function creates a NumPy array [1] of coefficients that follow a half-sine profile across
    the requested length. It uses `numpy.arange` [2] and `numpy.sin` [3] to place the samples. It
    acts on `lengthWindow` and returns the windowing coefficients.

    Parameters
    ----------
    lengthWindow : int
        Total length of the windowing function.
    device : Device = torch.device(device='cpu')
        PyTorch device for tensor allocation.

    Returns
    -------
    windowingFunction : WindowingFunction
        NumPy array of windowing coefficients following a half-sine shape.

    See Also
    --------
    `hunterHearsPy.optionalPyTorch.halfsineTensor`
        Tensor version of this windowing function.

    References
    ----------
    [1] NumPy `ndarray`.
        https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
    [2] NumPy `arange`.
        https://numpy.org/doc/stable/reference/generated/numpy.arange.html
    [3] NumPy `sin`.
        https://numpy.org/doc/stable/reference/generated/numpy.sin.html

    """
    device = device or torch.device(device='cpu')
    return _convertToTensor(lengthWindow, callableTarget=halfsine, device=device)

def tukeyTensor(lengthWindow: int, ratioTaper: float | None=None, device: Device | None=None, **keywordArguments: float) -> torch.Tensor:
    """Generate a Tukey windowing function with optional SciPy keyword overrides.

    This function creates a NumPy array [1] by delegating to SciPy's Tukey window implementation
    [2]. It accepts `lengthWindow`, `ratioTaper`, and extra keyword arguments, then returns the
    windowing coefficients.

    Parameters
    ----------
    lengthWindow : int
        Total length of the windowing function.
    ratioTaper : float | None = None
        Ratio of taper length to windowing-function length. The value must be between 0 and 1,
        inclusive.
    **keywordArguments : float
        Additional keyword arguments. `alpha` overrides `ratioTaper` when provided, matching SciPy's
        API.
    device : Device = torch.device(device='cpu')
        PyTorch device for tensor allocation.

    Returns
    -------
    windowingFunction : WindowingFunction
        NumPy array of Tukey windowing function coefficients.

    References
    ----------
    [1] NumPy `ndarray`.
        https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
    [2] SciPy Tukey window.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html

    """
    device = device or torch.device(device='cpu')
    return _convertToTensor(lengthWindow, ratioTaper, callableTarget=tukey, device=device, **keywordArguments)
