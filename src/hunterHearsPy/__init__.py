# ruff: noqa: D104
from __future__ import annotations

from hunterHearsPy.theTypes import (
	ArraySpectrograms as ArraySpectrograms, ArrayType as ArrayType, ArrayWaveforms as ArrayWaveforms,
	NormalizationReverter as NormalizationReverter, ParametersShortTimeFFT as ParametersShortTimeFFT, ParametersSTFT as ParametersSTFT,
	ParametersUniversal as ParametersUniversal, Spectrogram as Spectrogram, SpectrogramDtype as SpectrogramDtype, Waveform as Waveform,
	WaveformDtype as WaveformDtype, WaveformMetadata as WaveformMetadata, WindowingFunction as WindowingFunction,
	WindowingFunctionDtype as WindowingFunctionDtype, 个 as 个)

# isort: split
from hunterHearsPy.amplitude import normalizeArrayWaveforms as normalizeArrayWaveforms, normalizeWaveform as normalizeWaveform
from hunterHearsPy.autoRevert import moveToAxisOfOperation as moveToAxisOfOperation
from hunterHearsPy.clippingArrays import applyHardLimit as applyHardLimit, applyHardLimitComplexValued as applyHardLimitComplexValued
from hunterHearsPy.windowingFunctions import cosineWings as cosineWings, equalPower as equalPower, halfsine as halfsine, tukey as tukey

# isort: split
from hunterHearsPy.ioAudio import (
	getWaveformMetadata as getWaveformMetadata, lengthWindowingFunctionDEFAULT as lengthWindowingFunctionDEFAULT,
	loadSpectrograms as loadSpectrograms, loadWaveforms as loadWaveforms, parametersDEFAULT as parametersDEFAULT,
	parametersShortTimeFFTUniversal as parametersShortTimeFFTUniversal, parametersSTFTUniversal as parametersSTFTUniversal,
	readAudioFile as readAudioFile, resampleWaveform as resampleWaveform, setParametersUniversal as setParametersUniversal,
	spectrogramToWAV as spectrogramToWAV, stft as stft, universalDtypeSpectrogram as universalDtypeSpectrogram,
	universalDtypeWaveform as universalDtypeWaveform, waveformSpectrogramWaveform as waveformSpectrogramWaveform,
	windowingFunctionCallableDEFAULT as windowingFunctionCallableDEFAULT,
	windowingFunctionCallableUniversal as windowingFunctionCallableUniversal, writeWAV as writeWAV)

# isort: split
from contextlib import suppress

with suppress(ModuleNotFoundError):  # noqa: RUF067
	from hunterHearsPy.windowingFunctionsTensor import (
		callableReturnsNDArray as callableReturnsNDArray, cosineWingsTensor as cosineWingsTensor, equalPowerTensor as equalPowerTensor,
		halfsineTensor as halfsineTensor, tukeyTensor as tukeyTensor)
