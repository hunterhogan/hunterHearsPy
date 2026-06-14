# ruff: noqa: D100 D101 D103
from __future__ import annotations

from hunterHearsPy import OptionsAlign, ParametersUniversal, tukey, WaveformAxes
from hunterMakesPy import PackageSettings, raiseIfNone
from numpy import complex64, float32
from soundfile import dtype_str as soundfile_dtype_str
from typing import TYPE_CHECKING
import dataclasses

if TYPE_CHECKING:
	from hunterHearsPy import ParametersShortTimeFFT, ParametersSTFT
	from numpy.typing import DTypeLike

#================== Hardcoded =====================================================================

alignHARDCODED: OptionsAlign = 'start'
axisChannelHARDCODED: int = 0
axisWaveformTimeHARDCODED: int = 1
axisWaveformIndexingHARDCODED: int = 2
dtypeSpectrogramHARDCODED: DTypeLike = complex64
dtypeWaveformHARDCODED: DTypeLike = float32
# FailEarly A simple way to assure that the dtype string is consistent with the dtype object without using `assert`.
dtype_strHARDCODED: soundfile_dtype_str = raiseIfNone(dtypeWaveformHARDCODED.__name__
	if dtypeWaveformHARDCODED.__name__ in soundfile_dtype_str.__args__ else None)  # pyright: ignore[reportAssignmentType] # ty:ignore[invalid-assignment]
sampleRateHARDCODED: float = 44100

subtypeHARDCODED: str = 'FLOAT'

#================== Process yet to be invented to implement user settings =========================

align: OptionsAlign = alignHARDCODED
axisChannel: int = axisChannelHARDCODED
axisWaveformTime: int = axisWaveformTimeHARDCODED
axisWaveformIndexing: int = axisWaveformIndexingHARDCODED
dtype_str: soundfile_dtype_str = dtype_strHARDCODED
dtypeSpectrogram: DTypeLike = dtypeSpectrogramHARDCODED
dtypeWaveform: DTypeLike = dtypeWaveformHARDCODED
sampleRate: float = sampleRateHARDCODED

#================== "Data basket" à la `mapFolding` ===============================================

settingsPackage = PackageSettings('hunterHearsPy')

@dataclasses.dataclass(slots=True)
class UniversalParameters:
	align: OptionsAlign
	dtype_str: soundfile_dtype_str
	dtypeSpectrogram: DTypeLike
	dtypeWaveform: DTypeLike
	sampleRate: float

setting = UniversalParameters(
	align=align,
	dtype_str=dtype_str,
	dtypeSpectrogram=dtypeSpectrogram,
	dtypeWaveform=dtypeWaveform,
	sampleRate=sampleRate,
)

def getAxis() -> dict[str, WaveformAxes]:
	return dict(
		channel=WaveformAxes(number=axisChannel, size=0)
		, time=WaveformAxes(number=axisWaveformTime, size=0)
		, indexing=WaveformAxes(number=axisWaveformIndexing, size=0)
	)

#======= # TODO old system to be converted
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
