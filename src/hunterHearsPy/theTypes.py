# ruff: noqa: D101
"""Type definitions for audio signal processing and waveform analysis."""
from __future__ import annotations

from collections.abc import Callable
from numpy import complexfloating, dtype, floating, integer, ndarray, number
from soundfile import FileDescriptorOrPath as FileDescriptorOrPath  # noqa: TC002
from typing import Any, Literal, NamedTuple, TYPE_CHECKING, TypeAlias, TypedDict, TypeVar

if TYPE_CHECKING:
	from scipy.signal._short_time_fft import _FFTMode, _PadType, _ScaleTo

个 = TypeVar('个')
形floating = TypeVar('形floating', bound=floating[Any])
形ndarray = TypeVar('形ndarray', bound=ndarray[tuple[int, ...] | tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int], dtype[number]], covariant=True)
形Shape = TypeVar('形Shape', bound=tuple[int, ...] | tuple[int, int] | tuple[int, int, int] | tuple[int, int, int, int])

OptionsAlign: TypeAlias = Literal['center', 'start', 'stop']

#================== Waveform ======================================================================

WaveformDtype: TypeAlias = floating[Any] | integer[Any]
Waveform: TypeAlias = ndarray[tuple[int, int], dtype[WaveformDtype]]
"""A NumPy `ndarray` of audio waveform data with shape (channel, time); for mono audio, `channel` = 1."""

ArrayWaveforms: TypeAlias = ndarray[tuple[int, int, int], dtype[WaveformDtype]]
"""A NumPy `ndarray` containing `ndarray` of type `Waveform` indexed on the last axis: shape is (channel, time, `Waveform`)."""

class ArrayWaveformsShape(NamedTuple):
	a0: int
	a1: int
	a2: int

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

#================== ShortTimeFFT ==================================================================
WindowingFunctionDtype: TypeAlias = floating[Any]
WindowingFunction: TypeAlias = ndarray[tuple[int], dtype[WindowingFunctionDtype]]

class ParametersShortTimeFFT(TypedDict, total=False):
	win: WindowingFunction
	hop: int
	fs: int | float
	# *
	fft_mode: _FFTMode  # = "onesided"
	mfft: int | None  # = None

	dual_win: WindowingFunction | None  # = None
	scale_to: _ScaleTo | None  # = None
	phase_shift: int | None  # = 0

class ParametersSTFT(TypedDict, total=False):
	padding: _PadType
	axis: int

#==================================================================================================
# DEVELOPMENT refactoring. Below here, the objects have not yet been reviewed.

callableReturnsNDArray = TypeVar('callableReturnsNDArray', bound=Callable[..., WindowingFunction])
NormalizationReverter: TypeAlias = Callable[[Waveform], Waveform]
"""Function type for reversing normalization operations.

Type alias for callable objects that accept a normalized waveform and return the waveform restored to
its original amplitude scale.
"""
