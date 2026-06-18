# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportCallIssue=false
# pyright: reportUnknownVariableType=false
# ty:ignore[invalid-assignment]
from __future__ import annotations

from hunterHearsPy import ParametersShortTimeFFT, setting
from hunterMakesPy.parseParameters import defineConcurrencyLimit
from scipy.signal import ShortTimeFFT
from tqdm.auto import tqdm
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
		, CPUlimit: bool | float | int | None = None
		, **keywordArguments: Any
	) -> Waveform | ArrayWaveforms | Spectrogram | ArraySpectrograms:
	if inverse and not lengthWaveform:
		# TODO Save `arrayTarget` to a temp file in a temp dir because it might be the result of a
		# long computation. Print the pathFilename in the message.
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

	def mushroom(waveform: Waveform) -> Spectrogram:
		return workhorseSTFT.stft(x=waveform, padding=padding)

	def turtleShell(spectrogram: Spectrogram, lengthWaveform: int) -> Waveform:
		return workhorseSTFT.istft(S=spectrogram, k1=lengthWaveform)  # pyright: ignore[reportReturnType]

	if (indexingAxis is None) and (inverse is False):
		waveform: Waveform = arrayTarget
		return mushroom(waveform)
	elif (indexingAxis is None) and (inverse is True) and (lengthWaveform is not None):
		spectrogram: Spectrogram = arrayTarget
		return turtleShell(spectrogram, lengthWaveform)
	elif (indexingAxis is not None) and (inverse is False):
		max_workers: int = defineConcurrencyLimit(limit=CPUlimit)
		arrayWaveforms: ArrayWaveforms = arrayTarget
		arrayWaveforms = numpy.moveaxis(arrayWaveforms, indexingAxis, -1)
		index = 0
		arraySpectrograms: ArraySpectrograms = numpy.tile(mushroom(arrayWaveforms[..., index])[..., numpy.newaxis], arrayWaveforms.shape[-1])
		for index in range(1, arrayWaveforms.shape[-1]):
			arraySpectrograms[..., index] = mushroom(arrayWaveforms[..., index])
		return numpy.moveaxis(arraySpectrograms, -1, indexingAxis)
	elif (indexingAxis is not None) and (inverse is True) and (lengthWaveform is not None):
		max_workers: int = defineConcurrencyLimit(limit=CPUlimit)
		arrayTARGET: ArraySpectrograms = arrayTarget
		arrayTARGET = numpy.moveaxis(arrayTARGET, indexingAxis, -1)
		index = 0
		arrayTransformed: ArrayWaveforms = numpy.tile(turtleShell(arrayTARGET[..., index], lengthWaveform)[..., numpy.newaxis], arrayTARGET.shape[-1])
		for index in range(1, arrayTARGET.shape[-1]):
			arrayTransformed[..., index] = turtleShell(arrayTARGET[..., index], lengthWaveform)
		return numpy.moveaxis(arrayTransformed, -1, indexingAxis)
	else:
		message = "Invalid combination of `indexingAxis` and `inverse` parameters"
		raise ValueError(message)

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
