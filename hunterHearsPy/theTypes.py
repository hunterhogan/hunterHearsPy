# ruff:file-ignore[undocumented-public-class]
"""Type definitions for audio signal processing and waveform analysis.

I'm starting to feel that for _all_ identifiers in the package, there ought to be a strong correlation
between how often the identifier is imported in a `if TYPE_CHECKING:` block and whether the identifier
is defined in this module.

1. If it's almost never imported in any way, maybe it should be defined in the module that uses it.
2. If it's almost always imported in a `if TYPE_CHECKING:`, it should probably be defined in this
    module.
3. If it "feels" like a `type`, but is often not imported in a `if TYPE_CHECKING:`, then consider
    other module conventions, such as dataBaskets, semiotics, beDRY, and theSSOT.

(Even more meta: I feel like I am likely rediscovering ideas used by competent programmers.)

(More meta than meta: "rediscovering" is partially due to my extreme isolation, which is entirely due
to my untreated health conditions, and the extreme isolation prevents meaningful social experiences,
which is one of the major reasons that I hate my life.)
"""
from __future__ import annotations

from collections.abc import Callable
from numpy import complex64, complex128, dtype, floating, integer, ndarray, number
from soundfile import FileDescriptorOrPath as FileDescriptorOrPath
from typing import Any, Literal, TYPE_CHECKING, TypeAlias, TypedDict, TypeVar

if TYPE_CHECKING:
	from numpy.lib._arraypad_impl import _ModeKind
	from numpy.typing import DTypeLike
	from scipy.signal._short_time_fft import _FFTMode1, _PadType, _ScaleTo
	from soundfile import dtype_str as Options_dtype_str

#================== Waveform ======================================================================

WaveformFloatingDtype: TypeAlias = floating[Any]
WaveformFloating: TypeAlias = ndarray[tuple[int, int], dtype[WaveformFloatingDtype]]
"""NumPy `ndarray` of audio waveform data; shape (channel, time); mono audio, `channel` = 1."""
ArrayWaveformsFloating: TypeAlias = ndarray[tuple[int, int, int], dtype[WaveformFloatingDtype]]
"""NumPy `ndarray` of `WaveformFloating` indexed on the last axis: shape (channel, time, `WaveformFloating`)."""

WaveformDtype: TypeAlias = integer[Any] | WaveformFloatingDtype
Waveform: TypeAlias = ndarray[tuple[int, int], dtype[WaveformDtype]]
"""NumPy `ndarray` of audio waveform data; shape (channel, time); mono audio, `channel` = 1."""
ArrayWaveforms: TypeAlias = ndarray[tuple[int, int, int], dtype[WaveformDtype]]
"""NumPy `ndarray` of `Waveform` indexed on the last axis: shape (channel, time, `Waveform`)."""

#================== Spectrogram ===================================================================

SpectrogramDtype: TypeAlias = complex64 | complex128
Spectrogram: TypeAlias = ndarray[tuple[int, int, int], dtype[SpectrogramDtype]]
"""NumPy `ndarray` of complex-valued spectrogram data; shape (channel, frequencyBin, time); mono audio, `channel` = 1."""

ArraySpectrograms: TypeAlias = ndarray[tuple[int, int, int, int], dtype[SpectrogramDtype]]
"""NumPy `ndarray` of `Spectrogram` indexed on the last axis: shape (channel, frequencyBin, time, `Spectrogram`)."""

#================== WindowingFunction ==================================================================

WindowingFunctionDtype: TypeAlias = floating[Any]
WindowingFunction: TypeAlias = ndarray[tuple[int], dtype[WindowingFunctionDtype]]

#================== Function signatures ==================================================================

个 = TypeVar('个')
形floating = TypeVar('形floating', bound=floating[Any])
形ndarray = TypeVar('形ndarray', bound=ndarray[tuple[Any, ...], dtype[number]], covariant=True)
形Shape = TypeVar('形Shape', bound=tuple[Any, ...])
callableReturnsNDArray = TypeVar('callableReturnsNDArray', bound=Callable[..., WindowingFunction])

NormalizationReverter: TypeAlias = Callable[[Waveform], Waveform]
"""Function type for reversing normalization operations.

Type alias for callable objects that accept a normalized waveform and return the waveform restored to
its original amplitude scale.
"""

OptionsAlign: TypeAlias = Literal['center', 'start', 'stop']

class Parameters_loadWaveforms(TypedDict, total=False):
	align: OptionsAlign
	dtypeWaveform: DTypeLike
	dtype_str: Options_dtype_str
	sampleRate: float

class Parameters_loadSpectrograms(Parameters_loadWaveforms, total=False):
	align_pad_mode: _ModeKind
	dtypeSpectrogram: DTypeLike

# TODO except `padding`, the key names and types are duplicates of the field names and types in
# dataclass `ParametersShortTimeFFT`. I have NO idea how to make this DRY.
class Parameters_stft(TypedDict, total=False):
	dual_win: WindowingFunction | None
	fft_mode: _FFTMode1
	lengthFFT: int | None
	lengthHop: int
	padding: _PadType
	phase_shift: int | None
	sampleRate: int | float
	scale_to: _ScaleTo | None
	windowingFunction: WindowingFunction
