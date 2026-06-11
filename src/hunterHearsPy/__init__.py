# ruff: noqa: D104
from __future__ import annotations

from hunterHearsPy.theTypes import (
	ArraySpectrograms as ArraySpectrograms, ArrayTypeVariable as ArrayTypeVariable, ArrayWaveforms as ArrayWaveforms,
	callableReturnsNDArray as callableReturnsNDArray, FileDescriptorOrPath as FileDescriptorOrPath,
	NormalizationReverter as NormalizationReverter, ParametersShortTimeFFT as ParametersShortTimeFFT, ParametersSTFT as ParametersSTFT,
	ParametersUniversal as ParametersUniversal, Spectrogram as Spectrogram, SpectrogramDtype as SpectrogramDtype, Waveform as Waveform,
	WaveformDtype as WaveformDtype, WaveformMetadata as WaveformMetadata, WindowingFunction as WindowingFunction,
	WindowingFunctionDtype as WindowingFunctionDtype)

# isort: split
from hunterHearsPy.windowingFunctions import cosineWings as cosineWings, equalPower as equalPower, halfsine as halfsine, tukey as tukey

# isort: split
from contextlib import suppress

with suppress(ModuleNotFoundError):  # noqa: RUF067
	from hunterHearsPy.windowingFunctionsTensor import (
		cosineWingsTensor as cosineWingsTensor, equalPowerTensor as equalPowerTensor, halfsineTensor as halfsineTensor,
		tukeyTensor as tukeyTensor)

# isort: split
from hunterHearsPy.theSSOT import (
	parameters as parameters, parametersShortTimeFFTUniversal as parametersShortTimeFFTUniversal,
	parametersSTFTUniversal as parametersSTFTUniversal, setting as setting, universalDtypeSpectrogram as universalDtypeSpectrogram,
	universalDtypeWaveform as universalDtypeWaveform, windowingFunctionCallableUniversal as windowingFunctionCallableUniversal)

# isort: split
from hunterHearsPy._resample import resampleWaveform as resampleWaveform

# isort: split
from hunterHearsPy._io import readAudioFile as readAudioFile, writeWAV as writeWAV

# isort: split
from hunterHearsPy._fft import (
	spectrogramToWAV as spectrogramToWAV, stft as stft, waveformSpectrogramWaveform as waveformSpectrogramWaveform)

# isort: split
from hunterHearsPy.amplitude import normalizeArrayWaveforms as normalizeArrayWaveforms, normalizeWaveform as normalizeWaveform
from hunterHearsPy.clippingArrays import applyHardLimit as applyHardLimit, applyHardLimitComplexValued as applyHardLimitComplexValued

# isort: split
from hunterHearsPy.autoRevert import moveToAxisOfOperation as moveToAxisOfOperation

# isort: split
from hunterHearsPy._arrays import (
	getWaveformMetadata as getWaveformMetadata, loadSpectrograms as loadSpectrograms, loadWaveforms as loadWaveforms)
