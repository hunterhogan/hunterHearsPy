from __future__ import annotations

from hunterHearsPy import readAudioFile
from tests.conftest import assert_array_equal
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterHearsPy import Waveform
	from pathlib import Path
	from soundfile import dtype_str as Options_dtype_str

@pytest.mark.parametrize('sampleRateDesired', (44100, 48000))
@pytest.mark.parametrize(
	'pathFilename,dtype_str',
	(
		('Tone1000Hz_ch2_Hz44100_sec29_LUFS-23_s16.wav', 'int16'),
		('Speech_ch1_Hz44100_f32_sec60.wav', None),
		('Silence_ch1_Hz48000_s16_sec60.flac', 'int16'),
		('Music_chRsilent_Hz44100_s16_sec20.flac', 'int16'),
		('Music_ch2_Hz48000_s16_sec60_LUFS-20.wav', 'int16'),
		('Music_ch2_Hz44100_s16_peak0.wav', 'int16'),
		('Music_ch2_Hz44100_f32_sec20_RMS-20.wav', None),
	),
	indirect=['pathFilename'],
)
def test_readAudioFile(pathFilename: Path, sampleRateDesired: float, dtype_str: Options_dtype_str | None, expected: Waveform) -> None:
	actual: Waveform = readAudioFile(pathFilename=pathFilename, sampleRateDesired=sampleRateDesired, dtype_str=dtype_str)

	assert_array_equal(actual, expected, 'readAudioFile', pathFilename.name, sampleRateDesired=sampleRateDesired, dtype_str=dtype_str)
