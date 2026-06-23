from __future__ import annotations

from humpy_cytoolz.dicttoolz import keyfilter, merge
from hunterHearsPy import amplitudeIntegerToFloating, ParametersShortTimeFFT, setting, Translator
from scipy.signal import ShortTimeFFT
from typing import overload, TYPE_CHECKING
from typing_extensions import Unpack
import numpy

if TYPE_CHECKING:
	from collections.abc import Callable
	from hunterHearsPy import (
		ArraySpectrograms, ArrayWaveforms, ArrayWaveformsFloating, Parameters_stft, Spectrogram, Waveform, WaveformFloating)
	from pathlib import PurePath
	from scipy.signal._short_time_fft import _PadType

@overload  # stft 1 ndarray
def stft(arrayTarget: Waveform, *, lengthWaveform: int = 0, indexingAxis: int = -1, **keywordArguments: Unpack[Parameters_stft]) -> Spectrogram: ...
@overload  # stft many ndarray
def stft(arrayTarget: ArrayWaveforms, *, lengthWaveform: int = 0, indexingAxis: int = -1, **keywordArguments: Unpack[Parameters_stft]) -> ArraySpectrograms: ...
@overload  # istft 1 ndarray
def stft(arrayTarget: Spectrogram, *, lengthWaveform: int, indexingAxis: int = -1, **keywordArguments: Unpack[Parameters_stft]) -> Waveform: ...
@overload  # istft many ndarray
def stft(arrayTarget: ArraySpectrograms, *, lengthWaveform: int, indexingAxis: int = -1, **keywordArguments: Unpack[Parameters_stft]) -> ArrayWaveforms: ...
def stft(arrayTarget: Waveform | ArrayWaveforms | Spectrogram | ArraySpectrograms
		, *
		, lengthWaveform: int = 0
		, indexingAxis: int = -1
		, **keywordArguments: Unpack[Parameters_stft]
	) -> Waveform | ArrayWaveforms | Spectrogram | ArraySpectrograms:
	if numpy.issubdtype(arrayTarget.dtype, numpy.integer):
		if arrayTarget.ndim == 3:
			arrayFloating: ArrayWaveformsFloating = amplitudeIntegerToFloating(arrayTarget)
		if arrayTarget.ndim == 2:
			arrayFloating: WaveformFloating = amplitudeIntegerToFloating(arrayTarget)
	else:
		arrayFloating = arrayTarget
		if numpy.issubdtype(arrayTarget.dtype, numpy.complexfloating) and (lengthWaveform < 1):
			from hunterHearsPy._io import saveOnError  # noqa: PLC0415
			pathFilename: PurePath = saveOnError(arrayTarget)
			message: str = (
				"I did not receive `lengthWaveform`, so I could not perform the inverse STFT. "
				"I saved `arrayTarget` to a file in this computer's temporary directory so you might recover the data. "
				f"{arrayTarget.shape = }, {arrayTarget.dtype = }\n"
				f"{pathFilename = }"
			)
			raise ValueError(message)

	parametersShortTimeFFT = Translator(**ParametersShortTimeFFT(
		keyfilter(setting.ShortTimeFFT.keys().__contains__, merge(setting.ShortTimeFFT, keywordArguments))))  # pyright: ignore[reportArgumentType] # ty:ignore[invalid-argument-type]

	padding: _PadType = keywordArguments.get('padding', setting.padding)

	workhorseSTFT: ShortTimeFFT = ShortTimeFFT(**parametersShortTimeFFT.e733T)

	def mushroom(waveform: WaveformFloating) -> Spectrogram:
		return workhorseSTFT.stft(x=waveform, padding=padding)

	def turtleShell(spectrogram: Spectrogram, lengthWaveform: int) -> WaveformFloating:
		return workhorseSTFT.istft(S=spectrogram, k1=lengthWaveform)

	if arrayFloating.ndim == 2:
		arrayFloating: ArrayWaveformsFloating = numpy.expand_dims(arrayFloating, indexingAxis)
	elif (arrayTarget.ndim == 3) and (numpy.issubdtype(arrayTarget.dtype, numpy.complexfloating)):
		spectrogram: Spectrogram = arrayTarget
		return turtleShell(spectrogram, lengthWaveform)

	if (arrayFloating.ndim == 3) and (numpy.issubdtype(arrayFloating.dtype, numpy.floating)):
		arrayWaveforms: ArrayWaveformsFloating = arrayFloating
		arrayWaveforms = numpy.moveaxis(arrayWaveforms, indexingAxis, -1)
		index = 0
		arraySpectrograms: ArraySpectrograms = numpy.tile(mushroom(arrayWaveforms[..., index])[..., numpy.newaxis], arrayWaveforms.shape[-1])
		for index in range(1, arrayWaveforms.shape[-1]):
			arraySpectrograms[..., index] = mushroom(arrayWaveforms[..., index])
		arraySpectrograms = numpy.moveaxis(arraySpectrograms, -1, indexingAxis)
		if arraySpectrograms.shape[indexingAxis] == 1:
			arraySpectrograms = numpy.squeeze(arraySpectrograms, indexingAxis)
		return arraySpectrograms

	elif (arrayTarget.ndim == 4) and (numpy.issubdtype(arrayTarget.dtype, numpy.complexfloating)):
		arrayTARGET: ArraySpectrograms = arrayTarget
		arrayTARGET = numpy.moveaxis(arrayTARGET, indexingAxis, -1)
		index = 0
		arrayTransformed: ArrayWaveforms = numpy.tile(turtleShell(arrayTARGET[..., index], lengthWaveform)[..., numpy.newaxis], arrayTARGET.shape[-1])
		for index in range(1, arrayTARGET.shape[-1]):
			arrayTransformed[..., index] = turtleShell(arrayTARGET[..., index], lengthWaveform)
		return numpy.moveaxis(arrayTransformed, -1, indexingAxis)
	else:
		return arrayTarget

def waveformSpectrogramWaveform(callableNeedsSpectrogram: Callable[[Spectrogram], Spectrogram]) -> Callable[[Waveform], Waveform]:
	"""Decorate a spectrogram-processing callable to accept and return waveforms.

	You can use this function as a decorator when you have a function that transforms `Spectrogram`
	[1] data and you want a version that operates directly on `Waveform` [2] data. The returned
	function applies `stft` to convert the input `Waveform` [2] to a `Spectrogram` [1], calls
	`callableNeedsSpectrogram`, then applies inverse `stft` to convert the result back to a `Waveform`
	[2] of the original length.

	Parameters
	----------
	callableNeedsSpectrogram : Callable[[Spectrogram], Spectrogram]
		A function that accepts and returns a `Spectrogram` [1].

	Returns
	-------
	stft_istft : Callable[[Waveform], Waveform]
		A function that accepts a `Waveform` [2], converts it to a `Spectrogram` [1], applies
		`callableNeedsSpectrogram`, and returns the reconstructed `Waveform` [2] at the original
		length.

	Time Axis Assumption
	--------------------
	The inner function `stft_istft` assumes the time axis of the input `Waveform` [2] is the last axis
	(`-1`). This matches the `(channels, samples)` shape convention.

	References
	----------
	[1] `Spectrogram`

	[2] `Waveform`

	"""
	def stft_istft(waveform: Waveform) -> Waveform:
		axisTime = -1
		arrayTarget = stft(waveform)
		spectrogram = callableNeedsSpectrogram(arrayTarget)
		return stft(spectrogram, lengthWaveform=waveform.shape[axisTime])
	return stft_istft
