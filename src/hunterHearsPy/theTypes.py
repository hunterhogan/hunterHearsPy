"""Type definitions for audio signal processing and waveform analysis."""
from __future__ import annotations

from collections.abc import Callable
from numpy import complexfloating, dtype, float32, float64, floating, int16, int32, integer, ndarray
from soundfile import FileDescriptorOrPath as FileDescriptorOrPath  # noqa: TC002
from typing import Any, TYPE_CHECKING, TypeAlias, TypedDict, TypeVar
import numpy

if TYPE_CHECKING:
	from scipy.signal._short_time_fft import _FFTMode, _PadType, _ScaleTo

ArrayTypeVariable = TypeVar('ArrayTypeVariable', bound=ndarray[tuple[int, ...] | tuple[int, int], dtype[floating[Any]]], covariant=True)

# DEVELOPMENT refactoring. Below here, the objects have not yet been reviewed.
WaveformDtype: TypeAlias = float32 | float64
Waveform: TypeAlias = ndarray[tuple[int, int], dtype[WaveformDtype]]
"""A NumPy `ndarray` of audio waveform data with shape (channels, samples); for mono audio, `channels` = 1."""

WindowingFunctionDtype: TypeAlias = floating[Any]
WindowingFunction: TypeAlias = ndarray[tuple[int], dtype[WindowingFunctionDtype]]
callableReturnsNDArray = TypeVar('callableReturnsNDArray', bound=Callable[..., WindowingFunction])
ArrayWaveforms: TypeAlias = ndarray[tuple[int, int, int], dtype[WaveformDtype]]
"""A NumPy `ndarray` containing `ndarray` of type `Waveform` indexed on the last axis: shape is (channels, samples, `Waveform`)."""

SpectrogramDtype: TypeAlias = complexfloating[Any, Any]
Spectrogram: TypeAlias = ndarray[tuple[int, int, int], dtype[SpectrogramDtype]]
"""A NumPy `ndarray` of spectrogram data with shape (channels, frequency_bins, time_frames). For mono audio, `channels` = 1."""

ArraySpectrograms: TypeAlias = ndarray[tuple[int, int, int, int], dtype[SpectrogramDtype]]
"""A NumPy `ndarray` containing `ndarray` of type `Spectrogram` indexed on the last axis: shape is (channels, frequency_bins, time_frames, `Spectrogram`)."""

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

class WaveformMetadata(TypedDict):
	"""Metadata describing waveform file properties and processing state."""

	pathFilename: FileDescriptorOrPath
	lengthWaveform: int
	samplesLeading: int
	samplesTrailing: int

NormalizationReverter: TypeAlias = Callable[[Waveform], Waveform]
"""Function type for reversing normalization operations.

Type alias for callable objects that accept a normalized waveform and return the waveform restored to
its original amplitude scale.
"""
