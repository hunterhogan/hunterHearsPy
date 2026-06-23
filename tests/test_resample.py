from __future__ import annotations

from hunterHearsPy import resampleWaveform
from tests import assert_allclose, assert_array_equal
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterHearsPy import 形floating, 形Shape
	from numpy import dtype, float32, integer, ndarray
	from typing import Any

@pytest.mark.parametrize('axisTime', [pytest.param(-1)])
@pytest.mark.parametrize('pathFilename,sampleRateDesired'
	, indirect=['pathFilename']
	, argvalues=(
		pytest.param('Ambient_ch2_Hz48000_int16_sec60_peak-0.5dB.wav', 96000)
		, pytest.param('Ambient_ch2_Hz48000_int16_sec60_peak-0.5dB.wav', 48000)
		, pytest.param('Ambient_ch2_Hz48000_int16_sec60_peak-0.5dB.wav', 44100)
		, pytest.param('Discontinuous20-20000Hz_ch2_Hz48000_float32_sec20_peak-0.5dB.wav', 96000)
		, pytest.param('Discontinuous20-20000Hz_ch2_Hz48000_float32_sec20_peak-0.5dB.wav', 48000)
		, pytest.param('Discontinuous20-20000Hz_ch2_Hz48000_float32_sec20_peak-0.5dB.wav', 44100)
		, pytest.param('MusicNonVocal_ch2_Hz96000_float32_sec20_dBTP+0.5.wav', 96000)
		, pytest.param('MusicNonVocal_ch2_Hz96000_float32_sec20_dBTP+0.5.wav', 48000)
		, pytest.param('MusicNonVocal_ch2_Hz96000_float32_sec20_dBTP+0.5.wav', 44100)
		, pytest.param('Music_ch2_Hz44100_float32_sec20_peak0dBTP.wav', 48000)
		, pytest.param('Music_ch2_Hz44100_float32_sec20_peak0dBTP.wav', 44100)
		, pytest.param('Music_ch2_Hz44100_float32_sec20_peak0dBTP.wav', 32000)
		, pytest.param('PulsedA4_ch1_Hz44100_float32_sec20_peak-3.01dB.wav', 48000)
		, pytest.param('PulsedA4_ch1_Hz44100_float32_sec20_peak-3.01dB.wav', 44100)
		, pytest.param('PulsedA4_ch1_Hz44100_float32_sec20_peak-3.01dB.wav', 32000)
		, pytest.param('Speech_ch1_Hz16000_float32_sec20_peak-3.01dB.wav', 44100)
		, pytest.param('Speech_ch1_Hz16000_float32_sec20_peak-3.01dB.wav', 32000)
		, pytest.param('Speech_ch1_Hz16000_float32_sec20_peak-3.01dB.wav', 16000)
	)
)
def test_resampleWaveform(
	waveform: ndarray[形Shape, dtype[形floating]] | ndarray[形Shape, dtype[integer[Any]]]
	, sampleRateDesired: float
	, sampleRateSource: float
	, axisTime: int
	, rtol: float
	, atol: float
	, expected: ndarray[形Shape, dtype[形floating]] | ndarray[形Shape, dtype[float32]]
) -> None:
	actual: ndarray[形Shape, dtype[形floating]] | ndarray[形Shape, dtype[float32]] = resampleWaveform(waveform, sampleRateDesired, sampleRateSource, axisTime)

	if sampleRateDesired == sampleRateSource:
		assert_array_equal(actual, waveform, 'resampleWaveform', sampleRateDesired, sampleRateSource, axisTime)
	else:
		assert_allclose(actual, expected, rtol, atol, 'resampleWaveform', sampleRateDesired=sampleRateDesired, sampleRateSource=sampleRateSource, axisTime=axisTime)
