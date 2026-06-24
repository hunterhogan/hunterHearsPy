# ruff: noqa: D100, D101, D103
from __future__ import annotations

from hunterHearsPy import AxisMetadata, tukey
from hunterHearsPy.dataBaskets import ParametersShortTimeFFT
from hunterMakesPy import PackageSettings, raiseIfNone
from numpy import complex64, float32
from soundfile import dtype_str as Options_dtype_str
from typing import TYPE_CHECKING
import dataclasses

if TYPE_CHECKING:
	from hunterHearsPy.theTypes import OptionsAlign, WindowingFunction
	from numpy.lib._arraypad_impl import _ModeKind
	from numpy.typing import DTypeLike
	from scipy.signal._short_time_fft import _FFTMode1, _PadType, _ScaleTo

#================== Hardcoded =====================================================================

align_pad_modeHARDCODED: _ModeKind = 'reflect'
alignHARDCODED: OptionsAlign = 'start'

axisChannelHARDCODED: int = 0
axisWaveformTimeHARDCODED: int = 1
axisWaveformIndexingHARDCODED: int = 2

axisSpectrogramIndexingHARDCODED: int = 3

dtypeSpectrogramHARDCODED: DTypeLike = complex64
dtypeWaveformHARDCODED: DTypeLike = float32
#FailEarly A simple way to assure that the dtype string is consistent with the dtype object without using `assert`.
dtype_strHARDCODED: Options_dtype_str = raiseIfNone(dtypeWaveformHARDCODED.__name__
	if dtypeWaveformHARDCODED.__name__ in Options_dtype_str.__args__ else None)  # pyright: ignore[reportAssignmentType] # ty:ignore[invalid-assignment]
paddingHARDCODED: _PadType = 'even'
sampleRateHARDCODED: float = 44100
subtypeHARDCODED: str = 'FLOAT'

#------------------ ParametersShortTimeFFT --------------------------------------------------------

dual_winHARDCODED: WindowingFunction | None = None
fft_modeHARDCODED: _FFTMode1 = 'onesided'
lengthFFTHARDCODED: int = 2048
lengthHopHARDCODED: int = 512
phase_shiftHARDCODED: int | None = 0
scale_toHARDCODED: _ScaleTo | None = None
windowingFunctionHARDCODED: WindowingFunction = tukey(lengthHopHARDCODED * 2)

#================== Process yet to be invented to implement user settings =========================

align: OptionsAlign = alignHARDCODED
align_pad_mode: _ModeKind = align_pad_modeHARDCODED
axisChannel: int = axisChannelHARDCODED
axisWaveformIndexing: int = axisWaveformIndexingHARDCODED
axisWaveformTime: int = axisWaveformTimeHARDCODED
axisSpectrogramIndexing: int = axisSpectrogramIndexingHARDCODED
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
subtype: str = subtypeHARDCODED
windowingFunction: WindowingFunction = windowingFunctionHARDCODED

#================== "Data basket" à la `mapFolding` ===============================================

settingsPackage = PackageSettings('hunterHearsPy')

@dataclasses.dataclass(slots=True)
class UniversalParameters:
	align: OptionsAlign
	align_pad_mode: _ModeKind
	axisSpectrogramIndexing: int
	dtype_str: Options_dtype_str
	dtypeSpectrogram: DTypeLike
	dtypeWaveform: DTypeLike
	padding: _PadType
	sampleRate: float
	ShortTimeFFT: ParametersShortTimeFFT
	subtype: str

setting = UniversalParameters(
	align=align
	, align_pad_mode=align_pad_mode
	, axisSpectrogramIndexing=axisSpectrogramIndexing
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
		, windowingFunction=windowingFunction)
	, subtype=subtype
)

#------------------ Evolving idea for standardizing axes -------------------------------------------

def getAxis() -> dict[str, AxisMetadata]:
	return dict(
		channel=AxisMetadata(number=axisChannel, size=0)
		, time=AxisMetadata(number=axisWaveformTime, size=0)
		, indexing=AxisMetadata(number=axisWaveformIndexing, size=0)
	)
