# ruff: noqa: D104
from __future__ import annotations

from hunterHearsPy.theTypes import (
	ArraySpectrograms as ArraySpectrograms, ArrayWaveforms as ArrayWaveforms, ArrayWaveformsFloating as ArrayWaveformsFloating,
	ArrayWaveformsShape as ArrayWaveformsShape, callableReturnsNDArray as callableReturnsNDArray, E733TH4X0R as E733TH4X0R,
	FileDescriptorOrPath as FileDescriptorOrPath, NormalizationReverter as NormalizationReverter, OptionsAlign as OptionsAlign,
	Parameters_stft as Parameters_stft, ParametersShortTimeFFT as ParametersShortTimeFFT, Spectrogram as Spectrogram,
	SpectrogramDtype as SpectrogramDtype, Waveform as Waveform, WaveformAxes as WaveformAxes, WaveformDtype as WaveformDtype,
	WaveformFloating as WaveformFloating, WaveformFloatingDtype as WaveformFloatingDtype, WaveformMetadata as WaveformMetadata,
	WindowingFunction as WindowingFunction, WindowingFunctionDtype as WindowingFunctionDtype, 个 as 个, 形floating as 形floating,
	形ndarray as 形ndarray, 形Shape as 形Shape)

# isort: split
from hunterHearsPy.windowingFunctions import cosineWings as cosineWings, equalPower as equalPower, halfsine as halfsine, tukey as tukey

# isort: split
from contextlib import suppress

with suppress(ModuleNotFoundError):  # noqa: RUF067
	from hunterHearsPy.windowingFunctionsTensor import (
		cosineWingsTensor as cosineWingsTensor, equalPowerTensor as equalPowerTensor, halfsineTensor as halfsineTensor,
		tukeyTensor as tukeyTensor)

# isort: split
from hunterHearsPy.theSSOT import getAxis as getAxis, setting as setting

# isort: split
from hunterHearsPy.dataBaskets import Translator as Translator

# isort: split
from hunterHearsPy._resample import resampleWaveform as resampleWaveform
from hunterHearsPy.amplitude import (
	amplitudeIntegerToFloating as amplitudeIntegerToFloating, normalizeArrayWaveforms as normalizeArrayWaveforms,
	normalizeWaveform as normalizeWaveform)

# isort: split
from hunterHearsPy._fft import stft as stft, waveformSpectrogramWaveform as waveformSpectrogramWaveform

# isort: split
from hunterHearsPy._io import readAudioFile as readAudioFile, spectrogramToWAV as spectrogramToWAV, writeWAV as writeWAV

# isort: split
from hunterHearsPy.clippingArrays import applyHardLimit as applyHardLimit, applyHardLimitComplexValued as applyHardLimitComplexValued

# isort: split
from hunterHearsPy.autoRevert import moveToAxisOfOperation as moveToAxisOfOperation

# isort: split
from hunterHearsPy._arrays import (
	getWaveformMetadata as getWaveformMetadata, loadSpectrograms as loadSpectrograms, loadWaveforms as loadWaveforms)
