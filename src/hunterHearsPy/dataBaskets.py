# ruff: noqa: D100, D101, D102
from __future__ import annotations

from hunterHearsPy import E733TH4X0R, WindowingFunction
from typing import Literal, TYPE_CHECKING, TypeAlias
import dataclasses

if TYPE_CHECKING:
	from scipy.signal._short_time_fft import _ScaleTo
	_FFTMode1: TypeAlias = Literal["onesided", "onesided2X"]

@dataclasses.dataclass(slots=True)
class Translator:
	lengthHop: int
	sampleRate: int | float
	windowingFunction: WindowingFunction
	dual_win: WindowingFunction | None = None
	fft_mode: _FFTMode1 = 'onesided'
	lengthFFT: int | None = None
	phase_shift: int | None = 0
	scale_to: _ScaleTo | None = None

	@property
	def e733T(self) -> E733TH4X0R:
		return E733TH4X0R(
			dual_win=self.dual_win
			, fft_mode=self.fft_mode
			, fs=self.sampleRate
			, hop=self.lengthHop
			, mfft=self.lengthFFT
			, phase_shift=self.phase_shift
			, scale_to=self.scale_to
			, win=self.windowingFunction
		)
