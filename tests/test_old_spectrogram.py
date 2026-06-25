# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false
# ruff: noqa: DOC501
# ty:ignore[unresolved-attribute]
"""test_waveform or test_spectrogram? if a spectrogram is involved at any point, then test_spectrogram."""

from __future__ import annotations

from hunterHearsPy import loadSpectrograms, waveformSpectrogramWaveform
from pathlib import Path
from tests import messageTestFailure
from tests.oldSampleData import WaveformAndMetadata
from typing import Any, Final, TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable
	from hunterHearsPy.theTypes import Waveform
	from numpy.typing import NDArray

pathDataSamples = Path('tests/dataSamples/old')

expectedSpectrogramDimensions: Final[int] = 4

@pytest.fixture
def waveformDataStereo44kHz() -> WaveformAndMetadata:
	pathFilename = pathDataSamples / 'testSine2ch5sec.wav'
	return WaveformAndMetadata(pathFilename=pathFilename, LUFS=-23.0, sampleRate=44100.0, channelsTotal=2, ID='stereo44kHz')

@pytest.fixture
def listWaveformDataSameStereoShape() -> list[WaveformAndMetadata]:
	return [
		WaveformAndMetadata(pathFilename=pathDataSamples / 'testSine2ch5secCopy1.wav', LUFS=-23.0, sampleRate=44100.0, channelsTotal=2, ID='stereoCopy1'),
		WaveformAndMetadata(pathFilename=pathDataSamples / 'testSine2ch5secCopy2.wav', LUFS=-23.0, sampleRate=44100.0, channelsTotal=2, ID='stereoCopy2'),
		WaveformAndMetadata(pathFilename=pathDataSamples / 'testSine2ch5secCopy3.wav', LUFS=-23.0, sampleRate=44100.0, channelsTotal=2, ID='stereoCopy3'),
		WaveformAndMetadata(pathFilename=pathDataSamples / 'testSine2ch5secCopy4.wav', LUFS=-23.0, sampleRate=44100.0, channelsTotal=2, ID='stereoCopy4'),
	]

@pytest.fixture
def listPathFilenamesFromWaveformData(listWaveformDataSameStereoShape: list[WaveformAndMetadata]) -> list[Path]:
	"""Convert WaveformAndMetadata objects to path list for loadSpectrograms testing."""
	return [waveformData.pathFilename for waveformData in listWaveformDataSameStereoShape]

@pytest.mark.parametrize('sampleRateDesired', [22050, 44100, 48000])
def test_loadSpectrograms_acceptsSampleRateDesired(listPathFilenamesFromWaveformData: list[Path], sampleRateDesired: int) -> None:
	"""Test that loadSpectrograms accepts different target sample rates and produces valid spectrograms."""
	arraySpectrograms, dictionaryWaveformMetadata = loadSpectrograms(listPathFilenamesFromWaveformData, sampleRate=sampleRateDesired)

	expectedCountFiles = len(listPathFilenamesFromWaveformData)
	actualShape = arraySpectrograms.shape
	actualCountFiles = actualShape[-1]
	actualCountMetadata = len(dictionaryWaveformMetadata)

	assert actualCountFiles == expectedCountFiles, messageTestFailure(actualCountFiles, expectedCountFiles, 'loadSpectrograms', listPathFilenamesFromWaveformData, sampleRateDesired)
	assert actualCountMetadata == expectedCountFiles, messageTestFailure(
		actualCountMetadata, expectedCountFiles, 'loadSpectrograms metadata count', listPathFilenamesFromWaveformData, sampleRateDesired
	)
	assert len(actualShape) == expectedSpectrogramDimensions, messageTestFailure(
		len(actualShape), expectedSpectrogramDimensions, 'loadSpectrograms shape dimensions', listPathFilenamesFromWaveformData, sampleRateDesired
	)
	assert numpy.issubdtype(arraySpectrograms.dtype, numpy.complexfloating), messageTestFailure(
		arraySpectrograms.dtype, 'complex floating point type', 'loadSpectrograms dtype', listPathFilenamesFromWaveformData, sampleRateDesired
	)

def prototype_numpyAllClose(expected: NDArray[Any] | type[Exception], atol: float | None, rtol: float | None, functionTarget: Callable[..., Any], *arguments: Any, **keywordArguments: Any) -> None:
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
		assert actual == expected, messageTestFailure(messageActual, messageExpected, functionTarget.__name__, *arguments, **keywordArguments)
	else:
		if isinstance(expected, type):
			message = f'Expected an exception of type {expected.__name__}, but got a result'
			raise AssertionError(message)
		assert numpy.allclose(actual, expected, rtol, atol), messageTestFailure(actual, expected, functionTarget.__name__, *arguments, **keywordArguments)

"""Section: Spectrogram testing fixtures and parameters"""

def test_waveformSpectrogramWaveform_identityTransform(waveformDataStereo44kHz: WaveformAndMetadata) -> None:
	"""Test that waveformSpectrogramWaveform with identity function preserves waveforms."""
	waveformOriginal: Waveform = waveformDataStereo44kHz.waveform

	def identitySpectrogram(spectrogram: Any) -> Any:
		return spectrogram

	processor = waveformSpectrogramWaveform(identitySpectrogram)
	waveformProcessed = processor(waveformOriginal)

	prototype_numpyAllClose(waveformOriginal, 1e-2, 1e-2, lambda: waveformProcessed)
