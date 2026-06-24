# ruff: noqa: D100, D101, D102
from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING, TypedDict
import dataclasses

if TYPE_CHECKING:
	from hunterHearsPy.theTypes import ArraySpectrograms, ArrayWaveforms, FileDescriptorOrPath, WindowingFunction
	from scipy.signal._short_time_fft import _FFTMode1, _ScaleTo

class AxisMetadata(NamedTuple):
	number: int
	size: int

class SpectrogramsAndMetadata(NamedTuple):
	array: ArraySpectrograms
	metadata: dict[int, WaveformMetadata]

class E733TH4X0R(TypedDict, total=False):
	"""Low-semantic-value parameter names, used by elite hackers, of `scipy.signal.ShortTimeFFT`."""
	dual_win: WindowingFunction | None
	fft_mode: _FFTMode1
	fs: int | float
	hop: int
	mfft: int | None
	phase_shift: int | None
	scale_to: _ScaleTo | None
	win: WindowingFunction

@dataclasses.dataclass(slots=True)
# class Translator:
class ParametersShortTimeFFT:
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

class WaveformMetadata(TypedDict):
	"""Metadata describing waveform file properties and processing state."""

	channels: int
	lengthWaveform: int
	pathFilename: FileDescriptorOrPath
	# NOTE If the following values were assigned directly to a `slice` object, the slice object would
	# work as desired. https://docs.python.org/3/library/functions.html#slice Therefore, maintain this
	# functionality, and keep the semiotics aligned: `slice(start, stop)`.
	samplesStart: int
	samplesStop: int

class WaveformsAndMetadata(NamedTuple):
	array: ArrayWaveforms
	metadata: dict[int, WaveformMetadata]
