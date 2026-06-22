# ruff: noqa: D101
"""Type definitions for audio signal processing and waveform analysis."""
from __future__ import annotations

from collections.abc import Callable
from numpy import complex64, complex128, dtype, floating, integer, ndarray, number
from soundfile import FileDescriptorOrPath as FileDescriptorOrPath  # noqa: TC002
from typing import Any, Literal, NamedTuple, TYPE_CHECKING, TypeAlias, TypedDict, TypeVar

if TYPE_CHECKING:
	from scipy.signal._short_time_fft import _FFTMode1, _PadType, _ScaleTo

个 = TypeVar('个')
形floating = TypeVar('形floating', bound=floating[Any])
形ndarray = TypeVar('形ndarray', bound=ndarray[tuple[Any, ...], dtype[number]], covariant=True)
形Shape = TypeVar('形Shape', bound=tuple[Any, ...])

#================== Waveform ======================================================================

class ArraySpectrogramsShape(NamedTuple):
	a0: int
	a1: int
	a2: int
	a3: int

class ArrayWaveformsShape(NamedTuple):
	a0: int
	a1: int
	a2: int

class WaveformShape(NamedTuple):
	a0: int
	a1: int

OptionsAlign: TypeAlias = Literal['center', 'start', 'stop']

class AxisMetadata(NamedTuple):
	number: int
	size: int

WaveformFloatingDtype: TypeAlias = floating[Any]
WaveformFloating: TypeAlias = ndarray[tuple[int, int], dtype[WaveformFloatingDtype]]
"""A NumPy `ndarray` of audio waveform data with shape (channel, time); for mono audio, `channel` = 1."""
ArrayWaveformsFloating: TypeAlias = ndarray[tuple[int, int, int], dtype[WaveformFloatingDtype]]
"""A NumPy `ndarray` containing `ndarray` of type `WaveformFloating` indexed on the last axis: shape is (channel, time, `WaveformFloating`)."""

WaveformDtype: TypeAlias = integer[Any] | WaveformFloatingDtype
Waveform: TypeAlias = ndarray[tuple[int, int], dtype[WaveformDtype]]
"""A NumPy `ndarray` of audio waveform data with shape (channel, time); for mono audio, `channel` = 1."""
ArrayWaveforms: TypeAlias = ndarray[tuple[int, int, int], dtype[WaveformDtype]]
"""A NumPy `ndarray` containing `ndarray` of type `Waveform` indexed on the last axis: shape is (channel, time, `Waveform`)."""

NormalizationReverter: TypeAlias = Callable[[Waveform], Waveform]
"""Function type for reversing normalization operations.

Type alias for callable objects that accept a normalized waveform and return the waveform restored to
its original amplitude scale.
"""

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

SpectrogramDtype: TypeAlias = complex64 | complex128
Spectrogram: TypeAlias = ndarray[tuple[int, int, int], dtype[SpectrogramDtype]]
"""A NumPy `ndarray` of spectrogram data with shape (channel, frequency_bins, time). For mono audio, `channel` = 1."""

ArraySpectrograms: TypeAlias = ndarray[tuple[int, int, int, int], dtype[SpectrogramDtype]]
"""A NumPy `ndarray` containing `ndarray` of type `Spectrogram` indexed on the last axis: shape is (channel, frequency_bins, time, `Spectrogram`)."""

#================== ShortTimeFFT ==================================================================
WindowingFunctionDtype: TypeAlias = floating[Any]
WindowingFunction: TypeAlias = ndarray[tuple[int], dtype[WindowingFunctionDtype]]

callableReturnsNDArray = TypeVar('callableReturnsNDArray', bound=Callable[..., WindowingFunction])

class E733TH4X0R(TypedDict, total=False):
	dual_win: WindowingFunction | None
	fft_mode: _FFTMode1
	fs: int | float
	hop: int
	mfft: int | None
	phase_shift: int | None
	scale_to: _ScaleTo | None
	win: WindowingFunction

class ParametersShortTimeFFT(TypedDict, total=False):
	dual_win: WindowingFunction | None
	fft_mode: _FFTMode1
	lengthFFT: int | None
	lengthHop: int
	phase_shift: int | None
	sampleRate: int | float
	scale_to: _ScaleTo | None
	windowingFunction: WindowingFunction

class Parameters_stft(ParametersShortTimeFFT, total=False):
	padding: _PadType
