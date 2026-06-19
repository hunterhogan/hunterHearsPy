# ruff: noqa: D100, D101, D103, PLC2701
from __future__ import annotations

from hunterHearsPy import OptionsAlign, ParametersShortTimeFFT, tukey, WaveformAxes, WindowingFunction
from hunterMakesPy import PackageSettings, raiseIfNone
from numpy import complex64, float32
from soundfile import dtype_str as Options_dtype_str
from typing import Literal, TYPE_CHECKING, TypeAlias
import dataclasses

if TYPE_CHECKING:
	from numpy.typing import DTypeLike
	from scipy.signal._short_time_fft import _FFTMode, _PadType, _ScaleTo
	_FFTMode1: TypeAlias = Literal["onesided", "onesided2X"]

#================== Hardcoded =====================================================================

alignHARDCODED: OptionsAlign = 'start'
axisChannelHARDCODED: int = 0
axisWaveformTimeHARDCODED: int = 1
axisWaveformIndexingHARDCODED: int = 2
dtypeSpectrogramHARDCODED: DTypeLike = complex64
dtypeWaveformHARDCODED: DTypeLike = float32
#FailEarly A simple way to assure that the dtype string is consistent with the dtype object without using `assert`.
dtype_strHARDCODED: Options_dtype_str = raiseIfNone(dtypeWaveformHARDCODED.__name__
	if dtypeWaveformHARDCODED.__name__ in Options_dtype_str.__args__ else None)  # pyright: ignore[reportAssignmentType] # ty:ignore[invalid-assignment]
paddingHARDCODED: _PadType = 'even'
sampleRateHARDCODED: float = 44100

#------------------ ParametersShortTimeFFT ------------------------------------------------------------------------------

dual_winHARDCODED: WindowingFunction | None = None
fft_modeHARDCODED: _FFTMode1 = 'onesided'
lengthFFTHARDCODED: int = 2048
lengthHopHARDCODED: int = 512
phase_shiftHARDCODED: int | None = 0
scale_toHARDCODED: _ScaleTo | None = None
windowingFunctionHARDCODED: WindowingFunction = tukey(lengthHopHARDCODED * 2)

#------------------ TODO ------------------------------------------------------------------------------

subtypeHARDCODED: str = 'FLOAT'

#================== Process yet to be invented to implement user settings =========================

align: OptionsAlign = alignHARDCODED
axisChannel: int = axisChannelHARDCODED
axisWaveformIndexing: int = axisWaveformIndexingHARDCODED
axisWaveformTime: int = axisWaveformTimeHARDCODED
dtype_str: Options_dtype_str = dtype_strHARDCODED
dtypeSpectrogram: DTypeLike = dtypeSpectrogramHARDCODED
dtypeWaveform: DTypeLike = dtypeWaveformHARDCODED
dual_win: WindowingFunction | None = dual_winHARDCODED
fft_mode: _FFTMode1 = fft_modeHARDCODED
lengthFFT: int = lengthFFTHARDCODED
lengthHop: int = lengthHopHARDCODED
padding: _PadType = paddingHARDCODED
phase_shift: int | None = phase_shiftHARDCODED
sampleRate: float = sampleRateHARDCODED
scale_to: _ScaleTo | None = scale_toHARDCODED
windowingFunction: WindowingFunction = windowingFunctionHARDCODED

#================== "Data basket" à la `mapFolding` ===============================================

settingsPackage = PackageSettings('hunterHearsPy')

@dataclasses.dataclass(slots=True)
class UniversalParameters:
	align: OptionsAlign
	dtype_str: Options_dtype_str
	dtypeSpectrogram: DTypeLike
	dtypeWaveform: DTypeLike
	padding: _PadType
	sampleRate: float
	ShortTimeFFT: ParametersShortTimeFFT

setting = UniversalParameters(
	align=align
	, dtype_str=dtype_str
	, dtypeSpectrogram=dtypeSpectrogram
	, dtypeWaveform=dtypeWaveform
	, padding=padding
	, sampleRate=sampleRate
	, ShortTimeFFT=ParametersShortTimeFFT(
		dual_win=dual_win
		, fft_mode=fft_mode
		, lengthFFT=lengthFFT
		, lengthHop=lengthHop
		, phase_shift=phase_shift
		, sampleRate=sampleRate
		, scale_to=scale_to
		, windowingFunction=windowingFunction
))

#------------------ Evolving idea for standardizing axes -------------------------------------------

def getAxis() -> dict[str, WaveformAxes]:
	return dict(
		channel=WaveformAxes(number=axisChannel, size=0)
		, time=WaveformAxes(number=axisWaveformTime, size=0)
		, indexing=WaveformAxes(number=axisWaveformIndexing, size=0)
	)
