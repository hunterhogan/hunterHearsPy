from __future__ import annotations

from hunterHearsPy import ParametersShortTimeFFT, setting, WindowingFunctionDtype
from hunterMakesPy import zeroIndexed
from pathlib import Path
from scipy.signal import ShortTimeFFT
from typing import overload, TYPE_CHECKING
import numpy
import tempfile
import uuid

if TYPE_CHECKING:
	from collections.abc import Callable
	from hunterHearsPy import ArraySpectrograms, ArrayWaveforms, ArrayWaveformsFloating, Spectrogram, Waveform, WaveformFloating
	from scipy.signal._short_time_fft import _PadType
	from typing import Any

@overload  # stft 1 ndarray
def stft(arrayTarget: Waveform, *, lengthWaveform: None = None, indexingAxis: int = -1, **keywordArguments: Any) -> Spectrogram: ...
@overload  # stft many ndarray
def stft(arrayTarget: ArrayWaveforms, *, lengthWaveform: None = None, indexingAxis: int = -1, **keywordArguments: Any) -> ArraySpectrograms: ...
@overload  # istft 1 ndarray
def stft(arrayTarget: Spectrogram, *, lengthWaveform: int, indexingAxis: int = -1, **keywordArguments: Any) -> Waveform: ...
@overload  # istft many ndarray
def stft(arrayTarget: ArraySpectrograms, *, lengthWaveform: int, indexingAxis: int = -1, **keywordArguments: Any) -> ArrayWaveforms: ...
def stft(arrayTarget: Waveform | ArrayWaveforms | Spectrogram | ArraySpectrograms
		, *
		, lengthWaveform: int | None = None
		, indexingAxis: int = -1
		, **keywordArguments: Any
	) -> Waveform | ArrayWaveforms | Spectrogram | ArraySpectrograms:
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

	if numpy.issubdtype(arrayTarget.dtype, numpy.integer):
		integerInformation: numpy.iinfo[numpy.integer] = numpy.iinfo(arrayTarget.dtype.str)
		dtypeFloating: numpy.dtype[numpy.floating[Any]] = numpy.promote_types(arrayTarget.dtype, numpy.float32)
		arrayFloating: WaveformFloating | ArrayWaveformsFloating = numpy.astype(arrayTarget, dtypeFloating, copy=False)
		if integerInformation.min < 0:
			arrayFloating /= -integerInformation.min
		else:
			arrayFloating -= (integerInformation.max + zeroIndexed) / 2
			arrayFloating /= (integerInformation.max + zeroIndexed) / 2
	else:
		arrayFloating: WaveformFloating | ArrayWaveformsFloating = arrayTarget

	workhorseSTFT: ShortTimeFFT[WindowingFunctionDtype] = ShortTimeFFT(**parametersShortTimeFFT)

	def mushroom(waveform: WaveformFloating) -> Spectrogram:
		return workhorseSTFT.stft(x=waveform, padding=padding)

	def turtleShell(spectrogram: Spectrogram, lengthWaveform: int) -> WaveformFloating:
		return workhorseSTFT.istft(S=spectrogram, k1=lengthWaveform)  # pyright: ignore[reportReturnType]

	if arrayTarget.ndim == 2:
		waveform: WaveformFloating = arrayFloating
		return mushroom(waveform)
	elif (arrayTarget.ndim == 3) and (numpy.issubdtype(arrayTarget.dtype, numpy.complexfloating)):
		spectrogram: Spectrogram = arrayTarget
		if lengthWaveform is not None:
			return turtleShell(spectrogram, lengthWaveform)
		else:
			pathFilename: Path = Path(tempfile.mkdtemp(prefix='hunterHearsPy'), f"arrayTarget_{uuid.uuid4().hex}.npy").resolve()
			numpy.save(pathFilename, arrayTarget)
			message: str = (
				"I did not receive `lengthWaveform`, so I could not perform the inverse STFT. "
				f"I saved `arrayTarget` to a file in this computer's temporary directory so you might recover the data. {arrayTarget.shape = }, {arrayTarget.dtype = }\n"
				f"{pathFilename = }"
			)
			raise ValueError(message)
	elif (arrayTarget.ndim == 3) and (numpy.issubdtype(arrayTarget.dtype, numpy.floating)):
		arrayWaveforms: ArrayWaveformsFloating = arrayFloating
		arrayWaveforms = numpy.moveaxis(arrayWaveforms, indexingAxis, -1)
		index = 0
		arraySpectrograms: ArraySpectrograms = numpy.tile(mushroom(arrayWaveforms[..., index])[..., numpy.newaxis], arrayWaveforms.shape[-1])
		for index in range(1, arrayWaveforms.shape[-1]):
			arraySpectrograms[..., index] = mushroom(arrayWaveforms[..., index])
		return numpy.moveaxis(arraySpectrograms, -1, indexingAxis)
	elif (arrayTarget.ndim == 4) and (numpy.issubdtype(arrayTarget.dtype, numpy.complexfloating)):
		arrayTARGET: ArraySpectrograms = arrayTarget
		if lengthWaveform is not None:
			arrayTARGET = numpy.moveaxis(arrayTARGET, indexingAxis, -1)
			index = 0
			arrayTransformed: ArrayWaveforms = numpy.tile(turtleShell(arrayTARGET[..., index], lengthWaveform)[..., numpy.newaxis], arrayTARGET.shape[-1])
			for index in range(1, arrayTARGET.shape[-1]):
				arrayTransformed[..., index] = turtleShell(arrayTARGET[..., index], lengthWaveform)
			return numpy.moveaxis(arrayTransformed, -1, indexingAxis)
		else:
			pathFilename: Path = Path(tempfile.mkdtemp(prefix='hunterHearsPy'), f"arrayTarget_{uuid.uuid4().hex}.npy").resolve()
			numpy.save(pathFilename, arrayTarget)
			message: str = (
				"I did not receive `lengthWaveform`, so I could not perform the inverse STFT. "
				f"I saved `arrayTarget` to a file in this computer's temporary directory so you might recover the data. {arrayTarget.shape = }, {arrayTarget.dtype = }\n"
				f"{pathFilename = }"
			)
			raise ValueError(message)
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
