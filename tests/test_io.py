from __future__ import annotations

from hunterHearsPy import readAudioFile
from tests import assert_array_equal
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterHearsPy import Waveform
	from pathlib import Path
	from soundfile import dtype_str as Options_dtype_str

# TODO add `None` to `sampleRateDesired`.
@pytest.mark.parametrize('sampleRateDesired', (44100, 48000))
@pytest.mark.parametrize('pathFilename'
	, indirect=['pathFilename']
	, argvalues=(
		('Tone1000Hz_ch2_Hz44100_sec29_LUFS-23_int16.wav')
		, ('Speech_ch1_Hz44100_float32_sec60.wav')
		, ('Silence_ch1_Hz48000_int16_sec60.flac')
		, ('Music_chRsilent_Hz44100_int16_sec20.flac')
		, ('Music_ch2_Hz48000_int16_sec60_LUFS-20.wav')
		, ('Music_ch2_Hz44100_int16_peak0.wav')
		, ('Music_ch2_Hz44100_float32_sec20_RMS-20.wav')
	)
)
def test_readAudioFile(pathFilename: Path, sampleRateDesired: float, dtype_str: Options_dtype_str | None, expected: Waveform) -> None:
	actual: Waveform = readAudioFile(pathFilename, sampleRateDesired, dtype_str)

	assert_array_equal(actual, expected, 'readAudioFile', pathFilename.name, sampleRateDesired=sampleRateDesired, dtype_str=dtype_str)
