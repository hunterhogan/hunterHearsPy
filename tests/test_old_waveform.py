# pyright: reportUnknownMemberType=false
# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false
# pyright: reportUnknownVariableType=false
# ty:ignore[no-matching-overload]
# ruff: noqa: T201, PERF203, RUF076
from __future__ import annotations

from hunterHearsPy import getWaveformMetadata, loadWaveforms, readAudioFile, resampleWaveform, writeWAV
from pathlib import Path
from tests import assert_array_equal
from tests.oldSampleData import WaveformAndMetadata
from typing import Any, Final, TYPE_CHECKING
import io
import numpy
import pytest
import shutil
import soundfile
import uuid

if TYPE_CHECKING:
	from collections.abc import Generator

pathDataSamples = Path('tests/dataSamples/old')

"""Section: Audio file fixtures for testing readAudioFile, writeWAV, and related functions"""
pathTmpRoot: Path = pathDataSamples / 'tmp'

registerOfTemporaryFilesystemObjects: set[Path] = set()

def registrarRecordsTmpObject(path: Path) -> None:
	"""The registrar adds a tmp file to the register."""
	registerOfTemporaryFilesystemObjects.add(path)

def registrarDeletesTmpObjects() -> None:
	"""The registrar cleans up tmp files in the register."""
	for pathTmp in sorted(registerOfTemporaryFilesystemObjects, reverse=True):
		try:
			if pathTmp.is_file():
				pathTmp.unlink(missing_ok=True)
			elif pathTmp.is_dir():
				shutil.rmtree(pathTmp, ignore_errors=True)
		except Exception as ERRORmessage:
			print(f'Warning: Failed to clean up {pathTmp}: {ERRORmessage}')
			registerOfTemporaryFilesystemObjects.clear()

@pytest.fixture(scope='session', autouse=True)
def setupTeardownTmpObjects() -> Generator[None]:
	"""Auto-fixture to setup test data directories and cleanup after."""
	pathDataSamples.mkdir(exist_ok=True)
	pathTmpRoot.mkdir(exist_ok=True)
	yield
	registrarDeletesTmpObjects()

@pytest.fixture
def pathTmpTesting(request: pytest.FixtureRequest) -> Path:
	pathTmp = pathTmpRoot / str(uuid.uuid4().hex)
	pathTmp.mkdir(parents=True, exist_ok=False)

	registrarRecordsTmpObject(pathTmp)
	return pathTmp

@pytest.fixture
def pathFilenameTmpTesting(request: pytest.FixtureRequest) -> Path:
	try:
		extension: str = request.param
	except AttributeError:
		extension = '.txt'

	uuidHex: str = uuid.uuid4().hex
	subpath: str = uuidHex[0:-8]
	filenameStem: str = uuidHex[-8:None]

	pathFilenameTmp = Path(pathTmpRoot, subpath, filenameStem + extension)
	pathFilenameTmp.parent.mkdir(parents=True, exist_ok=False)

	registrarRecordsTmpObject(pathFilenameTmp)
	return pathFilenameTmp

@pytest.fixture
def mockTemporaryFiles(monkeypatch: pytest.MonkeyPatch, pathTmpTesting: Path) -> None:
	"""Mock all temporary filesystem operations to use pathTmpTesting."""
	monkeypatch.setattr('tempfile.mkdtemp', lambda *a, **k: str(pathTmpTesting))
	monkeypatch.setattr('tempfile.gettempdir', lambda: str(pathTmpTesting))
	monkeypatch.setattr('tempfile.mkstemp', lambda *a, **k: (0, str(pathTmpTesting)))

@pytest.fixture
def setupDirectoryStructure(pathTmpTesting: Path) -> Path:
	"""Create a complex directory structure for testing findRelativePath."""
	baseDirectory = pathTmpTesting / 'base'
	baseDirectory.mkdir()

	for subdir in ['dir1/subdir1', 'dir2/subdir2', 'dir3/subdir3']:
		(baseDirectory / subdir).mkdir(parents=True)

	(baseDirectory / 'dir1/file1.txt').touch()
	(baseDirectory / 'dir2/file2.txt').touch()

	return baseDirectory

@pytest.fixture
def waveformMono16kHz() -> WaveformAndMetadata:
	"""Fixture providing mono 16kHz waveform for readAudioFile testing."""
	pathFilename = pathDataSamples / 'testWooWooMono16kHz32integerClipping9sec.wav'
	return WaveformAndMetadata(pathFilename=pathFilename, LUFS=-23.0, sampleRate=16000.0, channelsTotal=1, ID='mono16kHz')

@pytest.fixture
def waveformStereo44kHz() -> WaveformAndMetadata:
	"""Fixture providing stereo 44.1kHz waveform for readAudioFile testing."""
	pathFilename = pathDataSamples / 'testSine2ch5sec.wav'
	return WaveformAndMetadata(pathFilename=pathFilename, LUFS=-23.0, sampleRate=44100.0, channelsTotal=2, ID='stereo44kHz')

@pytest.fixture
def waveformMono96kHz() -> WaveformAndMetadata:
	"""Fixture providing mono 96kHz waveform for resampleWaveform testing."""
	pathFilename = pathDataSamples / 'testParkMono96kHz32float12.1sec.wav'
	return WaveformAndMetadata(pathFilename=pathFilename, LUFS=-23.0, sampleRate=96000.0, channelsTotal=1, ID='mono96kHz')

@pytest.fixture
def waveformStereo48kHz() -> WaveformAndMetadata:
	"""Fixture providing stereo 48kHz waveform for testing."""
	pathFilename = pathDataSamples / 'testTrain2ch48kHz6.3sec.wav'
	return WaveformAndMetadata(pathFilename=pathFilename, LUFS=-23.0, sampleRate=48000.0, channelsTotal=2, ID='stereo48kHz')

@pytest.fixture
def listWaveformsSameStereoShape() -> list[WaveformAndMetadata]:
	"""Fixture providing multiple stereo waveforms with same shape for loadWaveforms testing."""
	basePath = pathDataSamples
	listWaveforms: list[WaveformAndMetadata] = []
	for indexCopy in [1, 2, 3, 4]:
		pathFilename = basePath / f'testSine2ch5secCopy{indexCopy}.wav'
		waveformData = WaveformAndMetadata(
			pathFilename=pathFilename, LUFS=-23.0, sampleRate=44100.0, channelsTotal=2, ID=f'stereoCopy{indexCopy}'
		)
		listWaveforms.append(waveformData)
	return listWaveforms

@pytest.fixture
def listWaveformsSameMonoShape() -> list[WaveformAndMetadata]:
	"""Fixture providing multiple mono waveforms with same shape for loadWaveforms testing."""
	basePath = pathDataSamples
	listWaveforms: list[WaveformAndMetadata] = []
	for indexCopy in [1, 2, 3]:
		pathFilename = basePath / f'testWooWooMono16kHz32integerClipping9secCopy{indexCopy}.wav'
		waveformData = WaveformAndMetadata(
			pathFilename=pathFilename, LUFS=-23.0, sampleRate=16000.0, channelsTotal=1, ID=f'monoCopy{indexCopy}'
		)
		listWaveforms.append(waveformData)
	return listWaveforms

@pytest.fixture
def pathFilenameVideoForErrorTesting() -> Path:
	"""Fixture providing video file path for testing error conditions."""
	return pathDataSamples / 'testVideo11sec.mkv'

@pytest.fixture
def pathFilenameNonexistentForErrorTesting() -> Path:
	"""Fixture providing nonexistent file path for testing error conditions."""
	return pathDataSamples / 'fileDoesNotExist.wav'

# Constants for test validation
CHANNELS_STEREO: Final[int] = 2
SAMPLE_RATE_DEFAULT: Final[int] = 44100
MESSAGE_EMPTY_FILE_LIST: Final[str] = ("I received `len(listPathFilenames) = 0`")

def test_readStereoFileDirectly(waveformStereo44kHz: WaveformAndMetadata) -> None:
	"""Test reading stereo files without modification."""
	waveformResult = readAudioFile(waveformStereo44kHz.pathFilename)
	assert waveformResult.shape[0] == CHANNELS_STEREO

@pytest.mark.parametrize('sampleRateDesired,tolerancePercent', [(22050, 5), (44100, 5), (48000, 5), (96000, 5)])
def test_resampleDuringRead(waveformStereo44kHz: WaveformAndMetadata, sampleRateDesired: int, tolerancePercent: int) -> None:
	"""Test resampling functionality during file reading."""
	secondsDuration = 5.0
	waveformResult = readAudioFile(waveformStereo44kHz.pathFilename, sampleRateDesired=sampleRateDesired)
	samplesExpected = int(sampleRateDesired * secondsDuration)
	samplesActual = waveformResult.shape[1]
	toleranceAbsolute = int(samplesExpected * tolerancePercent / 100)
	assert abs(samplesActual - samplesExpected) <= toleranceAbsolute

def test_loadMultipleStereoFiles(listWaveformsSameStereoShape: list[WaveformAndMetadata]) -> None:
	"""Test loading multiple stereo files into array format."""
	listPathFilenames = [waveformData.pathFilename for waveformData in listWaveformsSameStereoShape]
	arrayWaveformsResult = loadWaveforms(listPathFilenames)

	filesTotal = len(listWaveformsSameStereoShape)
	assert arrayWaveformsResult.shape[0] == CHANNELS_STEREO
	assert arrayWaveformsResult.shape[2] == filesTotal

def test_loadMixedMonoStereoFiles(waveformMono16kHz: WaveformAndMetadata, waveformStereo44kHz: WaveformAndMetadata) -> None:
	"""Test loading mixed mono and stereo files."""
	listPathFilenames = [waveformMono16kHz.pathFilename, waveformStereo44kHz.pathFilename]
	arrayWaveformsResult = loadWaveforms(listPathFilenames)

	filesTotal = 2
	assert arrayWaveformsResult.shape[0] == CHANNELS_STEREO  # All should be stereo
	assert arrayWaveformsResult.shape[2] == filesTotal

@pytest.mark.parametrize(
	'listPathFilenames,sampleRate,expectedMessage',
	[pytest.param([], SAMPLE_RATE_DEFAULT, MESSAGE_EMPTY_FILE_LIST, id='empty-list')],
)
def test_getWaveformMetadata(
	capsys: pytest.CaptureFixture[str], listPathFilenames: list[Path], sampleRate: int, expectedMessage: str
) -> None:
	"""Test the empty file-list metadata result and status message."""
	dictionaryWaveformMetadata, axis = getWaveformMetadata(listPathFilenames, sampleRate, 'start')

	capturedOutput = capsys.readouterr()
	assert capturedOutput.err.startswith(expectedMessage)
	assert not capturedOutput.out
	assert dictionaryWaveformMetadata == {}
	assert axis['channel'].size == 0
	assert axis['time'].size == 0
	assert axis['indexing'].size == 0

@pytest.mark.parametrize(
	'sampleRateSource,sampleRateDesired,factorExpected',
	[(16000, 44100, 2.75625), (44100, 22050, 0.5), (44100, 44100, 1.0), (96000, 48000, 0.5), (48000, 96000, 2.0)],
)
def test_resampleWithDifferentRates(
	waveformStereo44kHz: WaveformAndMetadata, sampleRateSource: int, sampleRateDesired: int, factorExpected: float
) -> None:
	"""Test resampling with various sample rate combinations."""
	waveformOriginal = waveformStereo44kHz.waveform
	waveformResampled = resampleWaveform(waveformOriginal, sampleRateDesired, sampleRateSource)

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
	waveformResampled = resampleWaveform(waveformOriginal, sampleRate, sampleRate)
	assert_array_equal(waveformResampled, waveformOriginal, 'resampleWaveform')

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
