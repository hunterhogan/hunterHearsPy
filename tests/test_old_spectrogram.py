# pyright: reportUnknownLambdaType=false
# pyright: reportUnknownArgumentType=false
# ruff: noqa: DOC201
"""test_waveform or test_spectrogram? if a spectrogram is involved at any point, then test_spectrogram."""
from __future__ import annotations

from hunterHearsPy import loadSpectrograms, readAudioFile, stft, waveformSpectrogramWaveform
from pathlib import Path
from tests.conftest import messageTestFailure, prototype_numpyAllClose, WaveformAndMetadata
from typing import Any, Final, TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from collections.abc import Callable

pathDataSamples = Path('tests/dataSamples/old')

expectedSpectrogramDimensions: Final[int] = 4

"""Section: Spectrogram testing fixtures and parameters"""

@pytest.fixture(params=[1024, 2048, 4096])
def lengthWindowingFunctionSTFT(request: pytest.FixtureRequest) -> int:
	"""Fixture providing different windowing function lengths for STFT testing."""
	return request.param

@pytest.fixture(params=[256, 512, 1024])
def lengthHopSTFT(request: pytest.FixtureRequest) -> int:
	"""Fixture providing different hop lengths for STFT testing."""
	return request.param

@pytest.fixture(params=[22050, 44100, 48000])
def sampleRateDesired(request: pytest.FixtureRequest) -> int:
	"""Fixture providing different target sample rates for spectrogram testing."""
	return request.param

@pytest.fixture
def waveformDataStereo44kHz() -> WaveformAndMetadata:
	"""Fixture providing stereo 44.1kHz waveform data for spectrogram testing."""
	pathFilename = pathDataSamples / 'testSine2ch5sec.wav'
	return WaveformAndMetadata(pathFilename=pathFilename, LUFS=-23.0, sampleRate=44100.0, channelsTotal=2, ID='stereo44kHz')

@pytest.fixture
def waveformDataMono16kHz() -> WaveformAndMetadata:
	"""Fixture providing mono 16kHz waveform data for spectrogram testing."""
	pathFilename = pathDataSamples / 'testWooWooMono16kHz32integerClipping9sec.wav'
	return WaveformAndMetadata(pathFilename=pathFilename, LUFS=-23.0, sampleRate=16000.0, channelsTotal=1, ID='mono16kHz')

@pytest.fixture
def listWaveformDataSameStereoShape() -> list[WaveformAndMetadata]:
	"""Fixture providing multiple stereo waveforms with same shape for spectrogram testing."""
	basePath = pathDataSamples
	return [
		WaveformAndMetadata(
			pathFilename=basePath / 'testSine2ch5secCopy1.wav', LUFS=-23.0, sampleRate=44100.0, channelsTotal=2, ID='stereoCopy1'
		)
		, WaveformAndMetadata(
			pathFilename=basePath / 'testSine2ch5secCopy2.wav', LUFS=-23.0, sampleRate=44100.0, channelsTotal=2, ID='stereoCopy2'
		)
		, WaveformAndMetadata(
			pathFilename=basePath / 'testSine2ch5secCopy3.wav', LUFS=-23.0, sampleRate=44100.0, channelsTotal=2, ID='stereoCopy3'
		)
		, WaveformAndMetadata(
			pathFilename=basePath / 'testSine2ch5secCopy4.wav', LUFS=-23.0, sampleRate=44100.0, channelsTotal=2, ID='stereoCopy4'
		)
	]

@pytest.fixture
def listPathFilenamesFromWaveformData(listWaveformDataSameStereoShape: list[WaveformAndMetadata]) -> list[Path]:
	"""Convert WaveformAndMetadata objects to path list for loadSpectrograms testing."""
	return [waveformData.pathFilename for waveformData in listWaveformDataSameStereoShape]

@pytest.mark.parametrize('sampleRateDesired', [22050, 44100, 48000])
def test_loadSpectrograms_acceptsSampleRateDesired(listPathFilenamesFromWaveformData: list[Path], sampleRateDesired: int) -> None:
	"""Test that loadSpectrograms accepts different target sample rates and produces valid spectrograms."""
	arraySpectrograms, dictionaryWaveformMetadata = loadSpectrograms(listPathFilenamesFromWaveformData, sampleRateDesired=sampleRateDesired)

	expectedCountFiles = len(listPathFilenamesFromWaveformData)
	actualShape = arraySpectrograms.shape
	actualCountFiles = actualShape[-1]
	actualCountMetadata = len(dictionaryWaveformMetadata)

	assert actualCountFiles == expectedCountFiles, messageTestFailure(
		actualCountFiles, expectedCountFiles, 'loadSpectrograms', listPathFilenamesFromWaveformData, sampleRateDesired
	)
	assert actualCountMetadata == expectedCountFiles, messageTestFailure(
		actualCountMetadata, expectedCountFiles, 'loadSpectrograms metadata count', listPathFilenamesFromWaveformData, sampleRateDesired
	)
	assert len(actualShape) == expectedSpectrogramDimensions, messageTestFailure(
		len(actualShape), expectedSpectrogramDimensions, 'loadSpectrograms shape dimensions', listPathFilenamesFromWaveformData, sampleRateDesired
	)
	assert numpy.issubdtype(arraySpectrograms.dtype, numpy.complexfloating), messageTestFailure(
		arraySpectrograms.dtype, 'complex floating point type', 'loadSpectrograms dtype', listPathFilenamesFromWaveformData, sampleRateDesired
	)

def test_loadSpectrograms_singleFile(waveformDataStereo44kHz: WaveformAndMetadata) -> None:
	"""Test loading a spectrogram from a single file."""
	sampleRateDesired = 44100
	listPathFilenameSingle = [waveformDataStereo44kHz.pathFilename]

	arraySpectrograms, dictionaryWaveformMetadata = loadSpectrograms(listPathFilenameSingle, sampleRateDesired=sampleRateDesired)

	waveform = readAudioFile(waveformDataStereo44kHz.pathFilename, sampleRateDesired)
	spectrogramExpected = stft(waveform, sampleRateDesired=sampleRateDesired)

	expectedShape = spectrogramExpected.shape
	actualShape = arraySpectrograms.shape[:-1]
	expectedCountFiles = 1
	actualCountFiles = arraySpectrograms.shape[-1]
	expectedCountMetadata = 1
	actualCountMetadata = len(dictionaryWaveformMetadata)

	assert actualShape == expectedShape, messageTestFailure(
		actualShape, expectedShape, 'loadSpectrograms single file shape', listPathFilenameSingle, sampleRateDesired
	)
	assert actualCountFiles == expectedCountFiles, messageTestFailure(
		actualCountFiles, expectedCountFiles, 'loadSpectrograms single file count', listPathFilenameSingle, sampleRateDesired
	)
	assert actualCountMetadata == expectedCountMetadata, messageTestFailure(
		actualCountMetadata, expectedCountMetadata, 'loadSpectrograms single file metadata count', listPathFilenameSingle, sampleRateDesired
	)

def test_loadSpectrograms_roundTripReconstructionAccuracy(waveformDataStereo44kHz: WaveformAndMetadata) -> None:
	"""Test that loadSpectrograms produces spectrograms that roundtrip through direct STFT operations."""
	sampleRateDesired = 44100
	listPathFilenameSingle = [waveformDataStereo44kHz.pathFilename]

	arraySpectrograms, _dictionaryWaveformMetadata = loadSpectrograms(listPathFilenameSingle, sampleRateDesired=sampleRateDesired)

	waveformOriginal = readAudioFile(waveformDataStereo44kHz.pathFilename, sampleRateDesired)
	spectrogramDirect = stft(waveformOriginal, sampleRateDesired=sampleRateDesired)

	expectedShape = spectrogramDirect.shape
	actualShape = arraySpectrograms.shape[:-1]
	expectedCountFiles = 1
	actualCountFiles = arraySpectrograms.shape[-1]

	assert actualShape == expectedShape, messageTestFailure(
		actualShape, expectedShape, 'loadSpectrograms roundtrip shape comparison', listPathFilenameSingle, sampleRateDesired
	)
	assert actualCountFiles == expectedCountFiles, messageTestFailure(
		actualCountFiles, expectedCountFiles, 'loadSpectrograms roundtrip file count', listPathFilenameSingle, sampleRateDesired
	)

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

	assert actual == expected, messageTestFailure(functionTarget.__name__, messageActual, messageExpected, *arguments, **keywordArguments)  # ty:ignore[unresolved-attribute]

def test_loadSpectrograms_rejectsEmptyInput() -> None:
	"""Test that loadSpectrograms raises TypeError for empty input."""
	standardizedEqualTo(TypeError, loadSpectrograms, [], 44100)

def test_stft_forwardTransform(waveformDataStereo44kHz: WaveformAndMetadata) -> None:
	"""Test that stft produces complex-valued spectrograms from real waveforms."""
	waveformSingle = waveformDataStereo44kHz.waveform

	spectrogram = stft(waveformSingle)

	actualDtype = spectrogram.dtype
	expectedComplexFloating = True
	actualComplexFloating = numpy.issubdtype(actualDtype, numpy.complexfloating)
	expectedNonEmpty = True
	actualNonEmpty = spectrogram.shape[0] > 0 and spectrogram.shape[1] > 0

	assert actualComplexFloating == expectedComplexFloating, messageTestFailure(
		actualDtype, 'complex floating point type', 'stft forward transform dtype', waveformSingle
	)
	assert actualNonEmpty == expectedNonEmpty, messageTestFailure(
		spectrogram.shape, 'non-empty spectrogram', 'stft forward transform shape', waveformSingle
	)

def test_stft_inverseTransform(waveformDataStereo44kHz: WaveformAndMetadata) -> None:
	"""Test that stft inverse transform reconstructs waveforms accurately."""
	waveformOriginal = waveformDataStereo44kHz.waveform

	spectrogram = stft(waveformOriginal)
	waveformReconstructed = stft(spectrogram, inverse=True, lengthWaveform=waveformOriginal.shape[1])

	prototype_numpyAllClose(waveformOriginal, 1e-2, 1e-2, lambda: waveformReconstructed)

@pytest.mark.parametrize('lengthWindowingFunctionSTFT,lengthHopSTFT', [(1024, 256), (2048, 512), (4096, 1024)])
def test_stft_acceptsSTFTParameters(
	waveformDataStereo44kHz: WaveformAndMetadata, lengthWindowingFunctionSTFT: int, lengthHopSTFT: int
) -> None:
	"""Test that stft accepts different windowing and hop parameters."""
	waveformSingle = waveformDataStereo44kHz.waveform
	sampleRate = waveformDataStereo44kHz.sampleRate

	spectrogram = stft(
		waveformSingle, sampleRate=sampleRate, lengthWindowingFunction=lengthWindowingFunctionSTFT, lengthHop=lengthHopSTFT
	)

	expectedNonEmpty = True
	actualNonEmpty = spectrogram.shape[0] > 0 and spectrogram.shape[1] > 0
	expectedComplexFloating = True
	actualComplexFloating = numpy.issubdtype(spectrogram.dtype, numpy.complexfloating)

	assert actualNonEmpty == expectedNonEmpty, messageTestFailure(
		actualNonEmpty, expectedNonEmpty, 'stft with custom parameters shape', waveformSingle, sampleRate=sampleRate, lengthWindowingFunction=lengthWindowingFunctionSTFT, lengthHop=lengthHopSTFT
	)
	assert actualComplexFloating == expectedComplexFloating, messageTestFailure(
		actualComplexFloating, expectedComplexFloating, 'stft with custom parameters dtype', waveformSingle, sampleRate=sampleRate, lengthWindowingFunction=lengthWindowingFunctionSTFT, lengthHop=lengthHopSTFT
	)

def test_stft_rejectsInverseWithoutLengthWaveform(waveformDataStereo44kHz: WaveformAndMetadata) -> None:
	"""Test that stft raises ValueError when inverse=True but lengthWaveform is not provided."""
	waveformSingle = waveformDataStereo44kHz.waveform
	spectrogram = stft(waveformSingle)

	standardizedEqualTo(ValueError, stft, spectrogram, inverse=True)

def test_waveformSpectrogramWaveform_identityTransform(waveformDataStereo44kHz: WaveformAndMetadata) -> None:
	"""Test that waveformSpectrogramWaveform with identity function preserves waveforms."""
	waveformOriginal = waveformDataStereo44kHz.waveform

	def identitySpectrogram(spectrogram: Any) -> Any:
		return spectrogram

	processor = waveformSpectrogramWaveform(identitySpectrogram)
	waveformProcessed = processor(waveformOriginal)

	prototype_numpyAllClose(waveformOriginal, 1e-2, 1e-2, lambda: waveformProcessed)
