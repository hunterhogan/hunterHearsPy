from __future__ import annotations

from functools import cache
from hunterHearsPy import readAudioFile, Waveform
from pathlib import Path
from typing import ClassVar

class WaveformAndMetadata:
	_cacheWaveforms: ClassVar[dict[Path, Waveform]] = {}

	def __init__(self, pathFilename: Path, LUFS: float, sampleRate: float, channelsTotal: int, ID: str) -> None:
		self.pathFilename: Path = pathFilename
		self.LUFS: float = LUFS
		self.sampleRate: float = sampleRate
		self.channelsTotal: int = channelsTotal
		self.ID: str = ID

	@property
	def waveform(self) -> Waveform:
		if self.pathFilename not in self._cacheWaveforms:
			self._cacheWaveforms[self.pathFilename] = readAudioFile(self.pathFilename, self.sampleRate)
		return self._cacheWaveforms[self.pathFilename]

pathDataSamples_labeled = Path('tests/dataSamples/labeled')

def ingestSampleData() -> list[WaveformAndMetadata]:
	"""Parse LUFS*.wav filenames and create WaveformData objects without loading waveforms."""  # noqa: DOC201
	listWaveformData: list[WaveformAndMetadata] = []
	for pathFilename in pathDataSamples_labeled.glob('LUFS*.wav'):
		LUFSAsStr, sampleRateAsStr, channelsTotalAsStr, ID = pathFilename.stem.split('_', maxsplit=3)
		LUFS = -float(LUFSAsStr[len('LUFS') :])
		sampleRate = float(sampleRateAsStr)
		channelsTotal = int(channelsTotalAsStr[len('ch') :])
		listWaveformData.append(WaveformAndMetadata(pathFilename, LUFS, sampleRate, channelsTotal, ID))
	return listWaveformData

@cache
def sampleData() -> list[WaveformAndMetadata]:
	return ingestSampleData()
