# ty:ignore[unresolved-attribute]
# ruff: noqa: DOC201, DOC501
from __future__ import annotations

from hunterHearsPy import readAudioFile
from pathlib import Path
from typing import Final, TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from collections.abc import Callable
	from hunterHearsPy import Waveform
	from numpy.typing import NDArray
	from typing import Any, ClassVar

pathDataSamples = Path('tests/dataSamples')

def messageTestFailure(function: str, actual: Any, expected: Any, *arguments: Any, **keywordArguments: Any) -> str:
	"""Format assertion message for any test comparison."""
	parameters: list[str] = list(map(repr, arguments))
	parameters.extend(f'{keyAndValue[0]}={keyAndValue[1]!r}' for keyAndValue in keywordArguments.items())
	return f'{function}({", ".join(parameters)}) = {actual!r}, but {expected = }'

#================== Old system: possibly refactor and keep ========================================

pathDataSamples_labeled = Path('tests/dataSamples/labeled')

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

def ingestSampleData() -> list[WaveformAndMetadata]:
	"""Parse LUFS*.wav filenames and create WaveformData objects without loading waveforms."""
	listWaveformData: list[WaveformAndMetadata] = []
	for pathFilename in pathDataSamples_labeled.glob('LUFS*.wav'):
		LUFSAsStr, sampleRateAsStr, channelsTotalAsStr, ID = pathFilename.stem.split('_', maxsplit=3)
		LUFS = -float(LUFSAsStr[len('LUFS') :])
		sampleRate = float(sampleRateAsStr)
		channelsTotal = int(channelsTotalAsStr[len('ch') :])
		listWaveformData.append(WaveformAndMetadata(pathFilename, LUFS, sampleRate, channelsTotal, ID))
	return listWaveformData

def sampleData() -> list[WaveformAndMetadata]:
	return ingestSampleData()

def standardizedEqualTo(expected: Any, functionTarget: Callable[..., Any], *arguments: Any, **keywordArguments: Any) -> None:
	"""Template for most tests to compare the actual outcome with the expected outcome, including expected errors."""
	if type(expected) == type[Exception]:  # noqa: E721
		messageExpected: str = expected.__name__
	else:
		messageExpected = expected

	try:
		messageActual = actual = functionTarget(*arguments, **keywordArguments)
	except Exception as actualError:
		messageActual: str = type(actualError).__name__
		actual = type(actualError)

	assert actual == expected, messageTestFailure(
		functionTarget.__name__, messageActual, messageExpected, *arguments, **keywordArguments
	)

def prototype_numpyAllClose(
	expected: NDArray[Any] | type[Exception]
	, atol: float | None
	, rtol: float | None
	, functionTarget: Callable[..., Any]
	, *arguments: Any
	, **keywordArguments: Any
) -> None:
	"""Template for tests using numpy.allclose comparison."""
	atolDEFAULT: Final[float] = 1e-7
	rtolDEFAULT: Final[float] = 1e-7

	if atol is None:
		atol = atolDEFAULT
	if rtol is None:
		rtol = rtolDEFAULT
	try:
		actual = functionTarget(*arguments, **keywordArguments)
	except Exception as actualError:
		messageActual: str = type(actualError).__name__
		actual = type(actualError)
		messageExpected = expected if isinstance(expected, type) else 'array-like result'
		assert actual == expected, messageTestFailure(
			functionTarget.__name__, messageActual, messageExpected, *arguments, **keywordArguments
		)
	else:
		if isinstance(expected, type):
			message = f'Expected an exception of type {expected.__name__}, but got a result'
			raise AssertionError(message)
		assert numpy.allclose(actual, expected, rtol, atol), messageTestFailure(
			functionTarget.__name__, actual, expected, *arguments, **keywordArguments
		)

def prototype_numpyArrayEqual(expected: NDArray[Any], functionTarget: Callable[..., Any], *arguments: Any, **keywordArguments: Any) -> None:
	"""Template for tests using numpy.array_equal comparison."""
	try:
		actual = functionTarget(*arguments, **keywordArguments)
	except Exception as actualError:
		messageActual: str = type(actualError).__name__
		actual = type(actualError)
		messageExpected = expected if isinstance(expected, type) else 'array-like result'
		assert actual == expected, messageTestFailure(
			functionTarget.__name__, messageActual, messageExpected, *arguments, **keywordArguments
		)
	else:
		assert numpy.array_equal(actual, expected), messageTestFailure(
			functionTarget.__name__, actual, expected, *arguments, **keywordArguments
		)
