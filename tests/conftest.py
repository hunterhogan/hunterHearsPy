# ty:ignore[invalid-assignment]
# ty:ignore[unresolved-attribute]
# pyright: reportUnknownLambdaType=false
# pyright: reportUnknownArgumentType=false
# ruff: noqa: DOC201, DOC501
from __future__ import annotations

from hunterHearsPy import readAudioFile
from numpy import float32
from pathlib import Path
from typing import TYPE_CHECKING
import numpy
import soundfile

if TYPE_CHECKING:
	from collections.abc import Callable
	from hunterHearsPy import Waveform
	from numpy import dtype, ndarray
	from numpy.typing import NDArray
	from typing import Any, ClassVar, Final

rtolDEFAULT: Final[float] = 1e-7

# SSOT for test data paths and filenames
pathDataSamples = Path('tests/dataSamples')
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
			if self.channelsTotal == 2:
				ImaWaveform: Waveform = readAudioFile(self.pathFilename, self.sampleRate)
			else:
				try:
					with soundfile.SoundFile(self.pathFilename) as readSoundFile:
						ImaSoundFile: ndarray[tuple[int, int], dtype[float32]] = readSoundFile.read(dtype='float32', always_2d=True).astype(
							float32
						)
				except soundfile.LibsndfileError as ERRORmessage:
					if 'System error' in str(ERRORmessage):
						message = f'File not found: {self.pathFilename}'
						raise FileNotFoundError(message) from ERRORmessage
					else:  # noqa: RET506
						raise
				ImaWaveform = ImaSoundFile.T
			self._cacheWaveforms[self.pathFilename] = ImaWaveform
		return self._cacheWaveforms[self.pathFilename]

def ingestSampleData() -> list[WaveformAndMetadata]:
	"""Parse LUFS*.wav filenames and create WaveformData objects without loading waveforms."""
	listWaveformData: list[WaveformAndMetadata] = []
	for pathFilename in pathDataSamples_labeled.glob('LUFS*.wav'):
		LUFSAsStr, sampleRateAsStr, channelsTotalAsStr, ID = pathFilename.stem.split('_', maxsplit=3)
		LUFS = -float(LUFSAsStr[len('LUFS') :])
		sampleRate = float(sampleRateAsStr)
		channelsTotal = int(channelsTotalAsStr[len('ch') :])
		listWaveformData.append(
			WaveformAndMetadata(pathFilename=pathFilename, LUFS=LUFS, sampleRate=sampleRate, channelsTotal=channelsTotal, ID=ID)
		)
	return listWaveformData

def sampleData() -> list[WaveformAndMetadata]:
	return ingestSampleData()

"""Section: Standardized assert statements and failure messages"""

def uniformTestFailureMessage(expected: Any, actual: Any, functionName: str, *arguments: Any, **keywordArguments: Any) -> str:
	"""Format assertion message for any test comparison."""
	listArgumentComponents: list[str] = [str(parameter) for parameter in arguments]
	listKeywordComponents: list[str] = [f'{key}={value}' for key, value in keywordArguments.items()]
	joinedArguments: str = ', '.join(listArgumentComponents + listKeywordComponents)

	return f'\nTesting: `{functionName}({joinedArguments})`\nExpected: {expected}\nGot: {actual}'

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

	assert actual == expected, uniformTestFailureMessage(
		messageExpected, messageActual, functionTarget.__name__, *arguments, **keywordArguments
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
		assert actual == expected, uniformTestFailureMessage(
			messageExpected, messageActual, functionTarget.__name__, *arguments, **keywordArguments
		)
	else:
		if isinstance(expected, type):
			message = f'Expected an exception of type {expected.__name__}, but got a result'
			raise AssertionError(message)
		assert numpy.allclose(actual, expected, rtol, atol), uniformTestFailureMessage(
			expected, actual, functionTarget.__name__, *arguments, **keywordArguments
		)

def prototype_numpyArrayEqual(expected: NDArray[Any], functionTarget: Callable[..., Any], *arguments: Any, **keywordArguments: Any) -> None:
	"""Template for tests using numpy.array_equal comparison."""
	try:
		actual = functionTarget(*arguments, **keywordArguments)
	except Exception as actualError:
		messageActual: str = type(actualError).__name__
		actual = type(actualError)
		messageExpected = expected if isinstance(expected, type) else 'array-like result'
		assert actual == expected, uniformTestFailureMessage(
			messageExpected, messageActual, functionTarget.__name__, *arguments, **keywordArguments
		)
	else:
		assert numpy.array_equal(actual, expected), uniformTestFailureMessage(
			expected, actual, functionTarget.__name__, *arguments, **keywordArguments
		)
