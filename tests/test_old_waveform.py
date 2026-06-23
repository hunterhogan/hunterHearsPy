# pyright: reportUnknownMemberType=false
# pyright: reportArgumentType=false
# pyright: reportCallIssue=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false
# pyright: reportUnknownVariableType=false
# ruff: noqa: T201, PERF203, RUF076
from __future__ import annotations

from hunterHearsPy import getWaveformMetadata, writeWAV
from pathlib import Path
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

# Constants for test validation
CHANNELS_STEREO: Final[int] = 2
SAMPLE_RATE_DEFAULT: Final[int] = 44100
MESSAGE_EMPTY_FILE_LIST: Final[str] = 'I received `len(listPathFilenames) = 0`'

@pytest.mark.parametrize('listPathFilenames,sampleRate,expectedMessage', [pytest.param([], SAMPLE_RATE_DEFAULT, MESSAGE_EMPTY_FILE_LIST, id='empty-list')])
def test_getWaveformMetadata(capsys: pytest.CaptureFixture[str], listPathFilenames: list[Path], sampleRate: int, expectedMessage: str) -> None:
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
