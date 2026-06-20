# ruff: noqa: DOC201, A002
from __future__ import annotations

from tests import pathDataSamples, pathDataSamplesExpected
from typing import TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from hunterHearsPy import Waveform, 个, 形ndarray
	from pathlib import Path
	from soundfile import dtype_str as Options_dtype_str
	from typing import Any

#================== Settings =====================================================================

@pytest.fixture()
def approx_rel(request: pytest.FixtureRequest) -> float:
	"""Return the relative tolerance for approximate comparisons."""
	return 1e-6

@pytest.fixture()
def approx_abs(request: pytest.FixtureRequest) -> float:
	"""Return the absolute tolerance for approximate comparisons."""
	return 1e-12

#================== Assert ========================================================================

def assert_approx(actual: 个, expected: 个, rel: float, abs: float, function: str, *arguments: Any, **keywordArguments: Any) -> None:
	assert actual == pytest.approx(expected, rel, abs, nan_ok=True), messageTestFailure(  # pyright: ignore[reportUnknownMemberType]
		actual, expected, function, *arguments, **keywordArguments)

def assert_array_equal(actual: 形ndarray, expected: 形ndarray, function: str, *arguments: Any, **keywordArguments: Any) -> None:
	"""Assert that two arrays are equal, and if not, raise an AssertionError with a detailed message."""
	assert numpy.array_equal(actual, expected), messageTestFailure_ndarray(actual, expected, function, *arguments, **keywordArguments)

def assertEqualTo(actual: 个, expected: 个, function: str, *arguments: Any, **keywordArguments: Any) -> None:
	"""Assert that two arrays are equal, and if not, raise an AssertionError with a detailed message."""
	assert actual == expected, messageTestFailure(actual, expected, function, *arguments, **keywordArguments)

#------------------ Messages ------------------------------------------------------------------------------

def messageTestFailure(actual: Any, expected: Any, function: str, *arguments: Any, **keywordArguments: Any) -> str:
	"""Format assertion message for any test comparison."""
	parameters: list[str] = list(map(repr, arguments))
	parameters.extend(f'{keyAndValue[0]}={keyAndValue[1]!r}' for keyAndValue in keywordArguments.items())
	return f'{function}({", ".join(parameters)}) = {actual!r}, but {expected = }.'

def messageTestFailure_ndarray(actual: 形ndarray, expected: 形ndarray, function: str, *arguments: Any, **keywordArguments: Any) -> str:
	parameters: list[str] = list(map(repr, arguments))
	parameters.extend(f'{keyAndValue[0]}={keyAndValue[1]!r}' for keyAndValue in keywordArguments.items())
	return (f'{function}({", ".join(parameters)}) = {actual.shape=},\t{actual.dtype=}, but {expected.shape=}, {expected.dtype=}.')

#================== Parameters ========================================================================

@pytest.fixture(
	params=[
		pytest.param(pathDataSamples / 'Tone1000Hz_ch2_44100Hz_29s_LUFS-23_s16.wav', id='Tone1000Hz_ch2_44100Hz_29s_LUFS-23_s16')
		, pytest.param(pathDataSamples / 'Speech_ch1_44100Hz_f32_60s.wav', id='Speech_ch1_44100Hz_f32_60s')
		, pytest.param(pathDataSamples / 'Silence_ch1_48000Hz_s16_60s.flac', id='Silence_ch1_48000Hz_s16_60s')
		, pytest.param(pathDataSamples / 'Music_chRsilent_44100Hz_s16_20s.flac', id='Music_chRsilent_44100Hz_s16_20s')
		, pytest.param(pathDataSamples / 'Music_ch2_48000Hz_s16_60s_LUFS-20.wav', id='Music_ch2_48000Hz_s16_60s_LUFS-20')
		, pytest.param(pathDataSamples / 'Music_ch2_44100Hz_s16_peak0.wav', id='Music_ch2_44100Hz_s16_peak0')
		, pytest.param(pathDataSamples / 'Music_ch2_44100Hz_f32_20s_RMS-20.wav', id='Music_ch2_44100Hz_f32_20s_RMS-20')
])
def pathFilename(request: pytest.FixtureRequest) -> Path:
	pathFilenameFromParameter: Path = request.param
	return pathFilenameFromParameter

@pytest.fixture
def dtype_str(pathFilename: Path) -> Options_dtype_str | None:
	dtypeStr: Options_dtype_str | None = None
	if 's16' in pathFilename.stem.split('_'):
		dtypeStr = 'int16'
	return dtypeStr

@pytest.fixture
def expected(pathFilename: Path, sampleRateDesired: float, dtype_str: Options_dtype_str | None) -> Waveform:
	pathFilenameExpected: Path = pathDataSamplesExpected / (
		f"readAudioFile__{pathFilename.stem}__sampleRateDesired{int(sampleRateDesired)}Hz__dtype{dtype_str}.npy"
	)
	return numpy.load(pathFilenameExpected, mmap_mode='r', allow_pickle=False)
