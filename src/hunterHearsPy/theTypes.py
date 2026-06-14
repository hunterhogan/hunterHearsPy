# ruff: noqa: D101
"""Type definitions for audio signal processing and waveform analysis."""
from __future__ import annotations

from collections.abc import Callable
from numpy import complexfloating, dtype, floating, integer, ndarray, number
from soundfile import FileDescriptorOrPath as FileDescriptorOrPath  # noqa: TC002
from typing import Any, Literal, NamedTuple, TYPE_CHECKING, TypeAlias, TypedDict, TypeVar

if TYPE_CHECKING:
	from scipy.signal._short_time_fft import _FFTMode, _PadType, _ScaleTo

ArrayTypeVariable = TypeVar('ArrayTypeVariable', bound=ndarray[tuple[int, ...] | tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int], dtype[number]], covariant=True)
ShapeTypeVariable = TypeVar('ShapeTypeVariable', bound=tuple[int, ...] | tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int])

OptionsAlign: TypeAlias = Literal['center', 'start', 'stop']

#================== Waveform ======================================================================

WaveformDtype: TypeAlias = floating[Any] | integer[Any]
Waveform: TypeAlias = ndarray[tuple[int, int], dtype[WaveformDtype]]
"""A NumPy `ndarray` of audio waveform data with shape (channel, time); for mono audio, `channel` = 1."""

ArrayWaveforms: TypeAlias = ndarray[tuple[int, int, int], dtype[WaveformDtype]]
"""A NumPy `ndarray` containing `ndarray` of type `Waveform` indexed on the last axis: shape is (channel, time, `Waveform`)."""

class WaveformAxes(NamedTuple):
	number: int
	size: int

class WaveformMetadata(TypedDict):
	"""Metadata describing waveform file properties and processing state."""

	channels: int
	lengthWaveform: int
	pathFilename: FileDescriptorOrPath
	# NOTE If the following values were assigned directly to a `slice` object, the slice object would
	# work as desired. https://docs.python.org/3/library/functions.html#slice Therefore, maintain this
	# functionality, and keep the semiotics aligned: `slice(start, stop)`.
	samplesStart: int
	samplesStop: int

#================== Spectrogram ===================================================================

SpectrogramDtype: TypeAlias = complexfloating[Any, Any]
Spectrogram: TypeAlias = ndarray[tuple[int, int, int], dtype[SpectrogramDtype]]
"""A NumPy `ndarray` of spectrogram data with shape (channel, frequency_bins, time). For mono audio, `channel` = 1."""

ArraySpectrograms: TypeAlias = ndarray[tuple[int, int, int, int], dtype[SpectrogramDtype]]
"""A NumPy `ndarray` containing `ndarray` of type `Spectrogram` indexed on the last axis: shape is (channel, frequency_bins, time, `Spectrogram`)."""

#==================================================================================================
# DEVELOPMENT refactoring. Below here, the objects have not yet been reviewed.

WindowingFunctionDtype: TypeAlias = floating[Any]
WindowingFunction: TypeAlias = ndarray[tuple[int], dtype[WindowingFunctionDtype]]
callableReturnsNDArray = TypeVar('callableReturnsNDArray', bound=Callable[..., WindowingFunction])

class ParametersSTFT(TypedDict, total=False):
	"""Optional parameters for Short-Time Fourier Transform operations."""

	padding: _PadType
	axis: int

class ParametersShortTimeFFT(TypedDict, total=False):
	"""Optional parameters for Short-Time FFT operations."""

	fft_mode: _FFTMode
	scale_to: _ScaleTo

class ParametersUniversal(TypedDict):
	"""Required parameters for universal audio processing operations."""

	lengthFFT: int
	lengthHop: int
	lengthWindowingFunction: int
	sampleRate: float
	windowingFunction: WindowingFunction

NormalizationReverter: TypeAlias = Callable[[Waveform], Waveform]
"""Function type for reversing normalization operations.

Type alias for callable objects that accept a normalized waveform and return the waveform restored to
its original amplitude scale.
"""
