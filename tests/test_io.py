from __future__ import annotations

from hunterHearsPy import readAudioFile
from tests.conftest import assert_array_equal
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterHearsPy import Waveform
	from pathlib import Path
	from soundfile import dtype_str as Options_dtype_str

@pytest.mark.parametrize(
	'sampleRateDesired', [pytest.param(44100, id='sampleRateDesired44100Hz'), pytest.param(48000, id='sampleRateDesired48000Hz')]
)
def test_readAudioFile(pathFilename: Path, sampleRateDesired: float, dtype_str: Options_dtype_str | None, expected: Waveform) -> None:
	actual: Waveform = readAudioFile(pathFilename=pathFilename, sampleRateDesired=sampleRateDesired, dtype_str=dtype_str)

	assert_array_equal(actual, expected, 'readAudioFile', pathFilename.name, sampleRateDesired=sampleRateDesired, dtype_str=dtype_str)
