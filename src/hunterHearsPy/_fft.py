# pyright: reportUnnecessaryComparison=false
# pyright: reportUnusedVariable=false
# ruff: noqa: FBT001
"""Read, write, resample, and transform audio waveforms between time and frequency domains.

You can use this module to load audio files into NumPy arrays, resample waveforms, convert
between waveforms and spectrograms using the Short-Time Fourier Transform, and write
waveforms back to WAV files. All audio is normalized to stereo, 32-bit float `Waveform`
arrays shaped `(channels, samples)`. All spectrograms are complex 64-bit float
`Spectrogram` arrays shaped `(channels, frequencies, frames)`.

Contents
--------
Functions
	getWaveformMetadata
		Retrieve metadata for a collection of audio waveform files.
	loadSpectrograms
		Load spectrograms from a list of audio files.
	loadWaveforms
		Load a list of audio files into a single stacked NumPy array.
	readAudioFile
		Read an audio file and return stereo waveform data as a NumPy array.
	resampleWaveform
		Resample a waveform array to a target sample rate.
	spectrogramToWAV
		Write a complex spectrogram to a WAV file.
	stft
		Perform Short-Time Fourier Transform or its inverse on waveform or spectrogram data.
	waveformSpectrogramWaveform
		Decorate a spectrogram-processing callable to accept and return waveforms.
	writeWAV
		Write a waveform array to a WAV file.

"""
from __future__ import annotations

from hunterHearsPy import parameters, parametersShortTimeFFTUniversal, parametersSTFTUniversal, windowingFunctionCallableUniversal, writeWAV
from math import ceil as ceiling, log2 as log_base2
from multiprocessing import set_start_method as multiprocessing_set_start_method
from scipy.signal import ShortTimeFFT
from typing import cast, overload, TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from collections.abc import Callable
	from hunterHearsPy import ArraySpectrograms, ArrayWaveforms, Spectrogram, Waveform, WindowingFunction
	from os import PathLike
	from typing import Any, BinaryIO, Literal

if __name__ == '__main__':
	multiprocessing_set_start_method('spawn')

@overload  # stft 1 ndarray
def stft(arrayTarget: Waveform, *, sampleRate: float | None = None, lengthHop: int | None = None, windowingFunction: WindowingFunction | None = None, lengthWindowingFunction: int | None = None, lengthFFT: int | None = None, inverse: Literal[False] = False, lengthWaveform: None = None, indexingAxis: None = None) -> Spectrogram: ...

@overload  # stft many ndarray
def stft(arrayTarget: ArrayWaveforms, *, sampleRate: float | None = None, lengthHop: int | None = None, windowingFunction: WindowingFunction | None = None, lengthWindowingFunction: int | None = None, lengthFFT: int | None = None, inverse: Literal[False] = False, lengthWaveform: None = None, indexingAxis: int = -1) -> ArraySpectrograms: ...

@overload  # istft 1 ndarray
def stft(arrayTarget: Spectrogram, *, sampleRate: float | None = None, lengthHop: int | None = None, windowingFunction: WindowingFunction | None = None, lengthWindowingFunction: int | None = None, lengthFFT: int | None = None, inverse: Literal[True], lengthWaveform: int, indexingAxis: None = None) -> Waveform: ...

@overload  # istft many ndarray
def stft(arrayTarget: ArraySpectrograms, *, sampleRate: float | None = None, lengthHop: int | None = None, windowingFunction: WindowingFunction | None = None, lengthWindowingFunction: int | None = None, lengthFFT: int | None = None, inverse: Literal[True], lengthWaveform: int, indexingAxis: int = -1) -> ArrayWaveforms: ...

def stft(arrayTarget: Waveform | ArrayWaveforms | Spectrogram | ArraySpectrograms
		, *
		, sampleRate: float | None = None
		, lengthHop: int | None = None
		, windowingFunction: WindowingFunction | None = None
		, lengthWindowingFunction: int | None = None
		, lengthFFT: int | None = None
		, inverse: bool = False
		, lengthWaveform: int | None = None
		, indexingAxis: int | None = None
		) -> Waveform | ArrayWaveforms | Spectrogram | ArraySpectrograms:
	"""Perform Short-Time Fourier Transform or its inverse on waveform or spectrogram data.

	You can use this function to convert a `Waveform` [1] to a `Spectrogram` [2] or reverse
	the transformation with `inverse=True`. Pass `ArrayWaveforms` [3] or
	`ArraySpectrograms` [4] with an `indexingAxis` to transform a batch of signals at once.
	All transform behavior is governed by `scipy.signal.ShortTimeFFT` [5].

	Four overloads determine the return type from `arrayTarget` and `inverse`:
	- `Waveform` [1] → `Spectrogram` [2] (single forward transform)
	- `ArrayWaveforms` [3] → `ArraySpectrograms` [4] (batch forward transform)
	- `Spectrogram` [2] → `Waveform` [1] (single inverse transform)
	- `ArraySpectrograms` [4] → `ArrayWaveforms` [3] (batch inverse transform)

	Parameters
	----------
	arrayTarget : Waveform | ArrayWaveforms | Spectrogram | ArraySpectrograms
		Input array for transformation.
	sampleRate : float | None = None
		Sample rate of the signal in Hz. Defaults to `44100` when `None`.
	lengthHop : int | None = None
		Number of samples between successive analysis frames. Defaults to `512` when `None`.
	windowingFunction : WindowingFunction | None = None
		Windowing function array [6]. When `None`, `windowingFunctionCallableUniversal` is
		called with `lengthWindowingFunction`, or the universal default is used.
	lengthWindowingFunction : int | None = None
		Length of the windowing function in samples. Used only when `windowingFunction` is
		`None`. Defaults to `1024` when `None`.
	lengthFFT : int | None = None
		Length of the FFT in samples. Defaults to `2048` or the next power of two ≥
		`lengthWindowingFunction` when `None`.
	inverse : bool = False
		When `True`, perform inverse STFT. When `False`, perform forward STFT.
	lengthWaveform : int | None = None
		Required output length in samples for inverse transform. Must be provided when
		`inverse` is `True`.
	indexingAxis : int | None = None
		Axis along which multiple signals are stacked. Use `None` for single-signal input.

	Returns
	-------
	arrayTransformed : Waveform | ArrayWaveforms | Spectrogram | ArraySpectrograms
		Transformed signal or batch of signals. Return type mirrors `arrayTarget` with
		forward and inverse swapped.

	Raises
	------
	ValueError
		When `inverse` is `True` and `lengthWaveform` is not provided.

	References
	----------
	[1] `Waveform`

	[2] `Spectrogram`

	[3] `ArrayWaveforms`

	[4] `ArraySpectrograms`

	[5] scipy.signal.ShortTimeFFT
		https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html

	[6] `WindowingFunction`

	"""
	if sampleRate is None:
		sampleRate = parameters['sampleRate']
	if lengthHop is None:
		lengthHop = parameters['lengthHop']

	if windowingFunction is None:
		if lengthWindowingFunction is not None and windowingFunctionCallableUniversal:
			windowingFunction = windowingFunctionCallableUniversal(lengthWindowingFunction)
		else:
			windowingFunction = parameters['windowingFunction']
		if lengthFFT is None:
			lengthFFTSherpa = parameters['lengthFFT']
			if lengthFFTSherpa >= windowingFunction.size:
				lengthFFT = lengthFFTSherpa

	if lengthFFT is None:
		lengthWindowingFunction = windowingFunction.size
		lengthFFT = 2 ** ceiling(log_base2(lengthWindowingFunction))

	if inverse and not lengthWaveform:
		message = "lengthWaveform must be specified for inverse transform"
		raise ValueError(message)

	stftWorkhorse = ShortTimeFFT(win=windowingFunction, hop=lengthHop, fs=sampleRate, mfft=lengthFFT, **parametersShortTimeFFTUniversal)

	def doTransformation(arrayInput: Waveform | Spectrogram, lengthWaveform: int | None, inverse: bool) -> Waveform | Spectrogram:
		if inverse:
			return cast('Waveform', stftWorkhorse.istft(S=arrayInput, k1=lengthWaveform))
		return cast('Spectrogram', stftWorkhorse.stft(x=arrayInput, **parametersSTFTUniversal))

	if indexingAxis is None:
		singleton: Waveform | Spectrogram = cast('Waveform | Spectrogram', arrayTarget)
		return doTransformation(singleton, lengthWaveform=lengthWaveform, inverse=inverse)
	else:
		arrayTARGET: ArrayWaveforms | ArraySpectrograms = cast('ArrayWaveforms | ArraySpectrograms', numpy.moveaxis(arrayTarget, indexingAxis, -1))
		index = 0
		arrayTransformed: ArrayWaveforms | ArraySpectrograms = cast('ArrayWaveforms | ArraySpectrograms', numpy.tile(doTransformation(cast('Waveform | Spectrogram', arrayTARGET[..., index]), lengthWaveform, inverse)[..., numpy.newaxis], arrayTARGET.shape[-1]))

		for index in range(1, arrayTARGET.shape[-1]):
			arrayTransformed[..., index] = doTransformation(cast('Waveform | Spectrogram', arrayTARGET[..., index]), lengthWaveform, inverse)

		return numpy.moveaxis(arrayTransformed, -1, indexingAxis)

def spectrogramToWAV(spectrogram: Spectrogram, pathFilename: str | PathLike[Any] | BinaryIO, lengthWaveform: int, sampleRate: float | None = None, **parametersSTFT: Any) -> None:
	"""Write a complex spectrogram to a WAV file by computing the inverse STFT.

	You can use this function to reconstruct a waveform from a `Spectrogram` [1] and save
	it directly to a WAV file. `spectrogramToWAV` calls `stft` with `inverse=True` to
	obtain the reconstructed `Waveform` [2], then passes it to `writeWAV`.

	Parameters
	----------
	spectrogram : Spectrogram
		Complex spectrogram to convert back to a waveform.
	pathFilename : str | PathLike[Any] | BinaryIO
		Destination path for the WAV file, or a binary stream.
	lengthWaveform : int
		Number of samples in the output waveform. The inverse STFT cannot recover the
		original length from the spectrogram alone, so `lengthWaveform` is required.
	sampleRate : float | None = None
		Sample rate for the output WAV file in Hz. Defaults to `44100` when `None`.
	**parametersSTFT : Any
		Keyword parameters forwarded to `stft`, such as `lengthWindowingFunction` and
		`lengthHop`.

	File Overwrite and Format
	-------------------------
	See `writeWAV` for file overwrite behavior and output format details.

	References
	----------
	[1] `Spectrogram`

	[2] `Waveform`

	"""
	if sampleRate is None:
		sampleRate = parameters['sampleRate']

	waveform: Waveform = stft(spectrogram, inverse=True, lengthWaveform=lengthWaveform, sampleRate=sampleRate, **parametersSTFT)
	writeWAV(pathFilename, waveform, sampleRate)

def waveformSpectrogramWaveform(callableNeedsSpectrogram: Callable[[Spectrogram], Spectrogram]) -> Callable[[Waveform], Waveform]:
	"""Decorate a spectrogram-processing callable to accept and return waveforms.

	You can use this function as a decorator when you have a function that transforms
	`Spectrogram` [1] data and you want a version that operates directly on `Waveform` [2]
	data. The returned function applies `stft` to convert the input `Waveform` [2] to a
	`Spectrogram` [1], calls `callableNeedsSpectrogram`, then applies inverse `stft` to
	convert the result back to a `Waveform` [2] of the original length.

	Parameters
	----------
	callableNeedsSpectrogram : Callable[[Spectrogram], Spectrogram]
		A function that accepts and returns a `Spectrogram` [1].

	Returns
	-------
	stft_istft : Callable[[Waveform], Waveform]
		A function that accepts a `Waveform` [2], converts it to a `Spectrogram` [1],
		applies `callableNeedsSpectrogram`, and returns the reconstructed `Waveform` [2]
		at the original length.

	Time Axis Assumption
	--------------------
	The inner function `stft_istft` assumes the time axis of the input `Waveform` [2] is
	the last axis (`-1`). This matches the `(channels, samples)` shape convention.

	References
	----------
	[1] `Spectrogram`

	[2] `Waveform`

	"""
	def stft_istft(waveform: Waveform) -> Waveform:
		axisTime = -1
		arrayTarget = stft(waveform)
		spectrogram = callableNeedsSpectrogram(arrayTarget)
		return stft(spectrogram, inverse=True, indexingAxis=None, lengthWaveform=waveform.shape[axisTime])
	return stft_istft
