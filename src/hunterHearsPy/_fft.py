from __future__ import annotations

from hunterHearsPy import FileDescriptorOrPath, ParametersShortTimeFFT, setting, writeWAV
from scipy.signal import ShortTimeFFT
from typing import overload, TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from collections.abc import Callable
	from hunterHearsPy import ArraySpectrograms, ArrayWaveforms, Spectrogram, Waveform
	from scipy.signal._short_time_fft import _PadType
	from typing import Any, Literal

@overload  # stft 1 ndarray
def stft(arrayTarget: Waveform, *, inverse: Literal[False] = False, lengthWaveform: None = None, indexingAxis: None = None, **keywordArguments: Any) -> Spectrogram: ...
@overload  # stft many ndarray
def stft(arrayTarget: ArrayWaveforms, *, inverse: Literal[False] = False, lengthWaveform: None = None, indexingAxis: int = -1, **keywordArguments: Any) -> ArraySpectrograms: ...
@overload  # istft 1 ndarray
def stft(arrayTarget: Spectrogram, *, inverse: Literal[True], lengthWaveform: int, indexingAxis: None = None, **keywordArguments: Any) -> Waveform: ...
@overload  # istft many ndarray
def stft(arrayTarget: ArraySpectrograms, *, inverse: Literal[True], lengthWaveform: int, indexingAxis: int = -1, **keywordArguments: Any) -> ArrayWaveforms: ...
def stft(arrayTarget: Waveform | ArrayWaveforms | Spectrogram | ArraySpectrograms
		, *
		, inverse: bool = False
		, lengthWaveform: int | None = None
		, indexingAxis: int | None = None
		, **keywordArguments: Any
	) -> Waveform | ArrayWaveforms | Spectrogram | ArraySpectrograms:
	if inverse and not lengthWaveform:
		message = "`lengthWaveform` must be specified for inverse transform"
		raise ValueError(message)

	parametersShortTimeFFT = ParametersShortTimeFFT(
		dual_win=keywordArguments.get('dual_win')
		, fft_mode=keywordArguments.get('fft_mode', setting.fft_mode)
		, hop=keywordArguments.get('lengthHop', setting.lengthHop)
		, mfft=keywordArguments.get('lengthFFT', setting.lengthFFT)
		, phase_shift=keywordArguments.get('phase_shift', 0)
		, fs=keywordArguments.get('sampleRate', setting.sampleRate)
		, scale_to=keywordArguments.get('scale_to')
		, win=keywordArguments.get('windowingFunction', setting.windowingFunction)
	)

	padding: _PadType = keywordArguments.get('padding', setting.padding)

	workhorseSTFT = ShortTimeFFT(**parametersShortTimeFFT)

	@overload
	def doTransformation(transformee: Waveform, lengthWaveform: None, *, inverse: Literal[False]) -> Spectrogram: ...
	@overload
	def doTransformation(transformee: Spectrogram, lengthWaveform: int, *, inverse: Literal[True]) -> Waveform: ...
	def doTransformation(transformee: Waveform | Spectrogram, lengthWaveform: int | None, *, inverse: bool) -> Waveform | Spectrogram:
		if inverse:
			return workhorseSTFT.istft(S=transformee, k1=lengthWaveform)
		return workhorseSTFT.stft(x=transformee, padding=padding)

	if indexingAxis is None:
		arrayOf1: Waveform | Spectrogram = arrayTarget
		return doTransformation(arrayOf1, lengthWaveform, inverse=inverse)
	else:
		arrayTARGET: ArrayWaveforms | ArraySpectrograms = numpy.moveaxis(arrayTarget, indexingAxis, -1)
		index = 0
		arrayTransformed: ArrayWaveforms | ArraySpectrograms = numpy.tile(doTransformation(arrayTARGET[..., index], lengthWaveform, inverse=inverse)[..., numpy.newaxis], arrayTARGET.shape[-1])

		for index in range(1, arrayTARGET.shape[-1]):
			arrayTransformed[..., index] = doTransformation(arrayTARGET[..., index], lengthWaveform, inverse=inverse)

		return numpy.moveaxis(arrayTransformed, -1, indexingAxis)

def spectrogramToWAV(spectrogram: Spectrogram, pathFilename: FileDescriptorOrPath, lengthWaveform: int, **parametersSTFT: Any) -> None:
	"""Write a complex spectrogram to a WAV file by computing the inverse STFT.

	You can use this function to reconstruct a waveform from a `Spectrogram` [1] and save
	it directly to a WAV file. `spectrogramToWAV` calls `stft` with `inverse=True` to
	obtain the reconstructed `Waveform` [2], then passes it to `writeWAV`.

	Parameters
	----------
	spectrogram : Spectrogram
		Complex spectrogram to convert back to a waveform.
	pathFilename : FileDescriptorOrPath
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
	waveform: Waveform = stft(spectrogram, inverse=True, lengthWaveform=lengthWaveform, indexingAxis=None, **parametersSTFT)
	sampleRate: float = parametersSTFT.get('sampleRate', setting.sampleRate)
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
