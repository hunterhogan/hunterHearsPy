"""Create PyTorch tensor windowing functions."""
from __future__ import annotations

from hunterHearsPy import cosineWings, equalPower, halfsine, tukey
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from hunterHearsPy.theTypes import callableReturnsNDArray
    from torch.types import Device
    from typing import Any

def _convertToTensor(*arguments: Any, callableTarget: callableReturnsNDArray, device: Device | None=None, dtype: torch.dtype | None=None, **keywordArguments: Any) -> torch.Tensor:
    arrayTarget = callableTarget(*arguments, **keywordArguments)
    if device is None:
        device = torch.device(device='cpu')
    return torch.tensor(data=arrayTarget, dtype=dtype or torch.float32, device=device)

def cosineWingsTensor(lengthSupport: int, ratioTaper: float=0.1, device: Device | None=None, dtype: torch.dtype | None=None) -> torch.Tensor:
    """Generate a cosine-tapered windowingFunction with a flat center and tapered ends.

    You can use this function to produce a 1-D windowingFunction of length `lengthSupport` whose ends
    are tapered with a cosine shape while the center remains at unity. The taper occupies `ratioTaper`
    of the total windowingFunction width and is split equally between the two ends.

    Parameters
    ----------
    lengthSupport : int
        Total length of the windowingFunction in samples.
    ratioTaper : float = 0.1
        Fraction of the total windowingFunction to taper. Must be in the closed interval [0, 1]. The
        computed taper length per end is `int(lengthSupport * ratioTaper / 2)`.
    device : Device = torch.device(device='cpu')
        PyTorch device for `Tensor`.
    dtype : torch.dtype = torch.float32
        PyTorch data type for `Tensor`.

    Returns
    -------
    windowingFunction : WindowingFunction
        1-D array of shape `(lengthSupport,)` containing values in [0, 1]. The centre region is 1.0
        and each end contains a cosine-shaped ramp from 0 → 1 (or 1 → 0) of length `lengthTaper`.
    """
    return _convertToTensor(lengthSupport, ratioTaper, callableTarget=cosineWings, device=device, dtype=dtype)

def equalPowerTensor(lengthSupport: int, ratioTaper: float=0.1, device: Device | None=None, dtype: torch.dtype | None=None) -> torch.Tensor:
    """Generate an equal-power windowingFunction suitable for crossfades.

    You can use this function to build a 1-D windowingFunction whose ends follow a square-root ramp.
    This produces an equal-power fade for crossfades, where amplitude ramps follow √(linear) shapes to
    preserve perceived loudness during mixing.

    Parameters
    ----------
    lengthSupport : int
        Total length of the windowingFunction in samples.
    ratioTaper : float = 0.1
        Fraction of the total windowingFunction to taper. Must be in the closed interval [0, 1]. The
        computed taper length per end is `int(lengthSupport * ratioTaper / 2)`.
    device : Device = torch.device(device='cpu')
        PyTorch device for `Tensor`.
    dtype : torch.dtype = torch.float32
        PyTorch data type for `Tensor`.

    Returns
    -------
    windowingFunction : WindowingFunction
        1-D array of shape `(lengthSupport,)` containing values in [0, 1]. The central region is 1.0
        and each end contains a √-shaped ramp of length `lengthTaper`.
    """
    return _convertToTensor(lengthSupport, ratioTaper, callableTarget=equalPower, device=device, dtype=dtype)

def halfsineTensor(lengthSupport: int, device: Device | None=None, dtype: torch.dtype | None=None) -> torch.Tensor:
    """Generate a half-sine windowingFunction of the requested length.

    This function returns a 1-D half-sine windowingFunction of length `lengthSupport`. The value at
    sample index `n` is `sin(π * (n + 0.5) / lengthSupport)`, producing a smoothly varying
    windowingFunction that starts and ends away from zero, commonly used in short-time analysis and
    overlap-add reconstruction.

    Parameters
    ----------
    lengthSupport : int
        Total length of the windowingFunction in samples.
    device : Device = torch.device(device='cpu')
        PyTorch device for `Tensor`.
    dtype : torch.dtype = torch.float32
        PyTorch data type for `Tensor`.

    Returns
    -------
    windowingFunction : WindowingFunction
        1-D array of shape `(lengthSupport,)` containing the half-sine values.

    References
    ----------
    [1] Short-Time Fourier Transform and Its Inverse.
    https://eeweb.engineering.nyu.edu/iselesni/EL713/STFT/stft_inverse.pdf
    """
    return _convertToTensor(lengthSupport, callableTarget=halfsine, device=device, dtype=dtype)

def tukeyTensor(lengthSupport: int, ratioTaper: float=0.1, device: Device | None=None, dtype: torch.dtype | None=None, **keywordArguments: float) -> torch.Tensor:
    """Generate a Tukey windowingFunction.

    You can use this function to obtain a tapered windowingFunction where the central region is
    constant and the ends are cosine-shaped. By default, the function uses `ratioTaper` as the Tukey
    `alpha` value. If an explicit `alpha` is provided via keyword arguments, that value is used
    instead of `ratioTaper`.

    Parameters
    ----------
    lengthSupport : int
        Total length of the windowingFunction in samples.
    ratioTaper : float = 0.1
        Default fraction of the windowingFunction to taper. Provided for API symmetry with other
        windowingFunction constructors.

    Other Parameters
    ----------------
    alpha : float = ratioTaper
        Equivalent to SciPy's `alpha` parameter. If provided, this overrides `ratioTaper`. Values are
        typically in the closed interval [0, 1].
    device : Device = torch.device(device='cpu')
        PyTorch device for `Tensor`.
    dtype : torch.dtype = torch.float32
        PyTorch data type for `Tensor`.

    Returns
    -------
    windowingFunction : WindowingFunction
        1-D array of shape `(lengthSupport,)`.
    """
    return _convertToTensor(lengthSupport, ratioTaper, callableTarget=tukey, device=device, dtype=dtype, **keywordArguments)
