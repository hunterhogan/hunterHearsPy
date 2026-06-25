from __future__ import annotations

from hunterHearsPy import amplitudeIntegerToFloating, stft
from tests import assert_allclose
from typing import TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from hunterHearsPy.theTypes import Parameters_stft, Spectrogram, Waveform

@pytest.mark.parametrize('keywordArguments', [pytest.param({}, id='keywordArguments~None')])
@pytest.mark.parametrize('indexingAxis', [pytest.param(-1)])
@pytest.mark.parametrize('lengthWaveform', [pytest.param(0)])
@pytest.mark.parametrize('pathFilename'
	, indirect=['pathFilename']
	, argvalues=(pytest.param('Ambient_ch2_Hz48000_int16_sec60_peak-0.5dB.wav')
		, pytest.param('Discontinuous20-20000Hz_ch2_Hz48000_float32_sec20_peak-0.5dB.wav')
		, pytest.param('Music_chRsilent_Hz44100_int16_sec20.flac')
		, pytest.param('MusicOnlyVocal_ch2_Hz44100_float32_sec20.wav')
		, pytest.param('Tone1000Hz_ch2_Hz44100_sec29_LUFS-23_int16.wav')
	)
)
def test_stft(arrayTarget: Waveform, lengthWaveform: int, indexingAxis: int, keywordArguments: Parameters_stft, rtol: float, atol: float, expected: Spectrogram) -> None:
	actual: Spectrogram = stft(arrayTarget, lengthWaveform=lengthWaveform, indexingAxis=indexingAxis, **keywordArguments)

	assert_allclose(actual, expected, rtol, atol, 'stft', arrayTarget, lengthWaveform=lengthWaveform, indexingAxis=indexingAxis, **keywordArguments)

	actualWaveform: Waveform = stft(actual, lengthWaveform=arrayTarget.shape[-1], indexingAxis=indexingAxis, **keywordArguments)
	if numpy.issubdtype(arrayTarget.dtype, numpy.integer):
		expectedWaveform = amplitudeIntegerToFloating(arrayTarget.copy())  # pyright: ignore[reportArgumentType]
	else:
		expectedWaveform = arrayTarget
	assert_allclose(actualWaveform, expectedWaveform, rtol, atol, 'stft', actual, lengthWaveform=arrayTarget.shape[-1], indexingAxis=indexingAxis, **keywordArguments)
