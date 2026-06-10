from __future__ import annotations

from hunterHearsPy import loadWaveforms, readAudioFile, resampleWaveform, writeWAV
from tests.conftest import prototype_numpyArrayEqual, standardizedEqualTo, WaveformAndMetadata
from typing import Any, Final, TYPE_CHECKING
import io
import numpy
import pytest
import soundfile

if TYPE_CHECKING:
	from pathlib import Path

# Constants for test validation
CHANNELS_STEREO: Final[int] = 2
SAMPLE_RATE_DEFAULT: Final[int] = 44100

def test_readMonoFileConvertsToStereo(waveformMono16kHz: WaveformAndMetadata) -> None:
	"""Test that mono files are automatically converted to stereo format."""
	waveformResult = readAudioFile(waveformMono16kHz.pathFilename)
	assert waveformResult.shape[0] == CHANNELS_STEREO

def test_readStereoFileDirectly(waveformStereo44kHz: WaveformAndMetadata) -> None:
	"""Test reading stereo files without modification."""
	waveformResult = readAudioFile(waveformStereo44kHz.pathFilename)
	assert waveformResult.shape[0] == CHANNELS_STEREO

@pytest.mark.parametrize('sampleRateTarget,tolerancePercent', [(22050, 5), (44100, 5), (48000, 5), (96000, 5)])
def test_resampleDuringRead(waveformStereo44kHz: WaveformAndMetadata, sampleRateTarget: int, tolerancePercent: int) -> None:
	"""Test resampling functionality during file reading."""
	secondsDuration = 5.0
	waveformResult = readAudioFile(waveformStereo44kHz.pathFilename, sampleRate=sampleRateTarget)
	samplesExpected = int(sampleRateTarget * secondsDuration)
	samplesActual = waveformResult.shape[1]
	toleranceAbsolute = int(samplesExpected * tolerancePercent / 100)
	assert abs(samplesActual - samplesExpected) <= toleranceAbsolute

def test_errorOnNonexistentFile(pathFilenameNonexistentForErrorTesting: Path) -> None:
	"""Test proper error handling for nonexistent files."""
	standardizedEqualTo(FileNotFoundError, readAudioFile, pathFilenameNonexistentForErrorTesting)

def test_errorOnVideoFile(pathFilenameVideoForErrorTesting: Path) -> None:
	"""Test proper error handling for unsupported file formats."""
	standardizedEqualTo(RuntimeError, readAudioFile, pathFilenameVideoForErrorTesting)

def test_loadMultipleStereoFiles(listWaveformsSameStereoShape: list[WaveformAndMetadata]) -> None:
	"""Test loading multiple stereo files into array format."""
	listPathFilenames = [waveformData.pathFilename for waveformData in listWaveformsSameStereoShape]
	arrayWaveformsResult = loadWaveforms(listPathFilenames)

	filesTotal = len(listWaveformsSameStereoShape)
	assert arrayWaveformsResult.shape[0] == CHANNELS_STEREO
	assert arrayWaveformsResult.shape[2] == filesTotal

def test_loadMultipleMonoFiles(listWaveformsSameMonoShape: list[WaveformAndMetadata]) -> None:
	"""Test loading multiple mono files with automatic stereo conversion."""
	listPathFilenames = [waveformData.pathFilename for waveformData in listWaveformsSameMonoShape]
	arrayWaveformsResult = loadWaveforms(listPathFilenames)

	filesTotal = len(listWaveformsSameMonoShape)
	assert arrayWaveformsResult.shape[0] == CHANNELS_STEREO  # Mono files should be converted to stereo
	assert arrayWaveformsResult.shape[2] == filesTotal

def test_loadMixedMonoStereoFiles(waveformMono16kHz: WaveformAndMetadata, waveformStereo44kHz: WaveformAndMetadata) -> None:
	"""Test loading mixed mono and stereo files."""
	listPathFilenames = [waveformMono16kHz.pathFilename, waveformStereo44kHz.pathFilename]
	arrayWaveformsResult = loadWaveforms(listPathFilenames)

	filesTotal = 2
	assert arrayWaveformsResult.shape[0] == CHANNELS_STEREO  # All should be stereo
	assert arrayWaveformsResult.shape[2] == filesTotal

def test_errorOnEmptyFileList() -> None:
	"""Test proper error handling for empty file lists."""
	standardizedEqualTo(ValueError, loadWaveforms, [])

@pytest.mark.parametrize(
	'sampleRateSource,sampleRateTarget,factorExpected',
	[(16000, 44100, 2.75625), (44100, 22050, 0.5), (44100, 44100, 1.0), (96000, 48000, 0.5), (48000, 96000, 2.0)],
)
def test_resampleWithDifferentRates(
	waveformStereo44kHz: WaveformAndMetadata, sampleRateSource: int, sampleRateTarget: int, factorExpected: float
) -> None:
	"""Test resampling with various sample rate combinations."""
	waveformOriginal = waveformStereo44kHz.waveform
	waveformResampled = resampleWaveform(waveformOriginal, sampleRateTarget, sampleRateSource)

	samplesExpected = int(waveformOriginal.shape[1] * factorExpected)
	samplesActual = waveformResampled.shape[1]
	assert samplesActual == samplesExpected

def test_resamplePreservesChannels(waveformStereo44kHz: WaveformAndMetadata) -> None:
	"""Test that resampling preserves channel count."""
	waveformOriginal = waveformStereo44kHz.waveform
	waveformResampled = resampleWaveform(waveformOriginal, 22050, 44100)
	assert waveformResampled.shape[0] == waveformOriginal.shape[0]

def test_resampleSameRateNoChange(waveformStereo44kHz: WaveformAndMetadata) -> None:
	"""Test that identical sample rates produce no change."""
	waveformOriginal = waveformStereo44kHz.waveform
	sampleRate = 44100
	prototype_numpyArrayEqual(waveformOriginal, resampleWaveform, waveformOriginal, sampleRate, sampleRate)

@pytest.mark.parametrize(
	'testCase',
	[
		{'channelsTotal': 1, 'samplesTotal': 1000, 'description': 'mono audio', 'shapeExpectedSoundfile': (1000,)},
		{'channelsTotal': 2, 'samplesTotal': 1000, 'description': 'stereo audio', 'shapeExpectedSoundfile': (1000, 2)},
	],
)
def test_writeAndVerifyContent(pathFilenameTmpTesting: Path, testCase: dict[str, Any]) -> None:
	"""Test writing WAV files and verifying their contents match expectations."""
	waveformTest = numpy.full((testCase['channelsTotal'], testCase['samplesTotal']), 0.5, dtype=numpy.float32)
	writeWAV(pathFilenameTmpTesting, waveformTest)

	waveformRead, sampleRateRead = soundfile.read(pathFilenameTmpTesting)

	assert sampleRateRead == SAMPLE_RATE_DEFAULT

	assert waveformRead.shape == testCase['shapeExpectedSoundfile']

	if testCase['channelsTotal'] == 1:
		numpy.testing.assert_allclose(waveformRead, waveformTest.flatten())
	else:
		numpy.testing.assert_allclose(waveformRead, waveformTest.T)

def test_writeCreatesDirectories(pathTmpTesting: Path) -> None:
	"""Test that writeWAV creates necessary directory structure."""
	pathFilenameNested = pathTmpTesting / 'nested' / 'directories' / 'test.wav'
	waveformTest = numpy.ones((2, 1000), dtype=numpy.float32)
	writeWAV(pathFilenameNested, waveformTest)
	assert pathFilenameNested.exists()

def test_writeOverwritesExistingFile(pathFilenameTmpTesting: Path) -> None:
	"""Test that writeWAV properly overwrites existing files."""
	waveformFirst = numpy.ones((2, 1000), dtype=numpy.float32)
	waveformSecond = numpy.zeros((2, 1000), dtype=numpy.float32)

	writeWAV(pathFilenameTmpTesting, waveformFirst)
	writeWAV(pathFilenameTmpTesting, waveformSecond)

	waveformRead, _sampleRateRead = soundfile.read(pathFilenameTmpTesting)
	numpy.testing.assert_allclose(waveformRead.T, waveformSecond)

def test_writeToBinaryStream() -> None:
	"""Test writing audio data to a binary stream object."""
	waveformTest = numpy.full((2, 1000), 0.25, dtype=numpy.float32)
	streamBinary = io.BytesIO()
	writeWAV(streamBinary, waveformTest)

	# Verify data was written to the stream
	bytesWritten = streamBinary.getvalue()
	assert len(bytesWritten) > 0
