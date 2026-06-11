# ruff: noqa: D100 D101
from __future__ import annotations

from hunterHearsPy import FileDescriptorOrPath, ParametersUniversal, tukey
from hunterMakesPy import PackageSettings, raiseIfNone
from numpy import complex64, float32
from typing import Literal, TYPE_CHECKING
import dataclasses

if TYPE_CHECKING:
	from hunterHearsPy import ParametersShortTimeFFT, ParametersSTFT
	from numpy.typing import DTypeLike

settingsPackage = PackageSettings('hunterHearsPy')

# TODO I want reliably consistent waveforms, spectrograms, and transformations.
# Therefore, I want one package, this package, to manage those objects.
# Within the package, I need a single source of truth for the settings of those objects.
# I need good default universal settings.
# But if I want a different universal setting, I need an easy way to change the universal setting.
# Until I learn how to make an easy way to change the universal settings, I will use the HARDCODED
# system as a placeholder.
# Finally, I want to be able to override the universal settings as needed on a per-call basis.

# No librosa.
# Adapters for torch, but the work is done by NumPy, scipy, or other packages I trust.

@dataclasses.dataclass(slots=True)
class UniversalParameters:
	dtype_str: Literal['float32', 'float64']
	dtypeSpectrogram: DTypeLike
	dtypeWaveform: DTypeLike
	sampleRate: float

#================== Hardcoded =====================================================================

dtypeSpectrogramHARDCODED: DTypeLike = complex64
dtypeWaveformHARDCODED: DTypeLike = float32
dtype_strHARDCODED: Literal['float32', 'float64'] = raiseIfNone(dtypeWaveformHARDCODED.__name__  # FailEarly A simple way to assure that the dtype string is consistent with the dtype object without using assert.
	if dtypeWaveformHARDCODED.__name__ in {'float32', 'float64'} else None)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportAssignmentType] # ty:ignore[invalid-assignment]
sampleRateHARDCODED: float = 44100

#================== Process yet to be invented to implement user settings =========================

dtype_str: Literal['float32', 'float64'] = dtype_strHARDCODED
dtypeSpectrogram: DTypeLike = dtypeSpectrogramHARDCODED
dtypeWaveform: DTypeLike = dtypeWaveformHARDCODED
sampleRate: float = sampleRateHARDCODED

#================== "Data basket" à la `mapFolding` ===============================================

setting = UniversalParameters(
	dtype_str=dtype_str,
	dtypeSpectrogram=dtypeSpectrogram,
	dtypeWaveform=dtypeWaveform,
	sampleRate=sampleRate,
)

#======= # TODO old system to be converted
universalDtypeWaveform = float32
universalDtypeSpectrogram = complex64
parametersShortTimeFFTUniversal: ParametersShortTimeFFT = {'fft_mode': 'onesided'}
parametersSTFTUniversal: ParametersSTFT = {'padding': 'even', 'axis': -1}

lengthWindowingFunctionDEFAULT = 1024
windowingFunctionCallableDEFAULT = tukey
parametersDEFAULT = ParametersUniversal(
	lengthFFT=2048
	, lengthHop=512
	, lengthWindowingFunction=lengthWindowingFunctionDEFAULT
	, sampleRate=44100
	, windowingFunction=windowingFunctionCallableDEFAULT(lengthWindowingFunctionDEFAULT),
)

windowingFunctionCallableUniversal = windowingFunctionCallableDEFAULT

parameters: ParametersUniversal = parametersDEFAULT
