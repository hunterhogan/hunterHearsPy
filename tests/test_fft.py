from __future__ import annotations

from hunterHearsPy import amplitudeIntegerToFloating, getAxis, stft
from tests import assert_allclose, assertEqualTo
from typing import TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from hunterHearsPy.theTypes import ArraySpectrograms, ArrayWaveforms, Parameters_stft, Spectrogram, Waveform
	from pathlib import Path

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
def test_stft(arrayTarget: Waveform, lengthWaveform: int, indexingAxis: int, keywordArguments: Parameters_stft, pathFilename: Path, rtol: float, atol: float, expected: Spectrogram) -> None:
	actual: Spectrogram = stft(arrayTarget, lengthWaveform=lengthWaveform, indexingAxis=indexingAxis, **keywordArguments)

	assert_allclose(actual, expected, rtol, atol, 'stft', arrayTarget, lengthWaveform=lengthWaveform, indexingAxis=indexingAxis, **keywordArguments)

	actualWaveform: Waveform = stft(actual, lengthWaveform=arrayTarget.shape[-1], indexingAxis=indexingAxis, **keywordArguments)
	if numpy.issubdtype(arrayTarget.dtype, numpy.integer):
		expectedWaveform = amplitudeIntegerToFloating(arrayTarget.copy())  # pyright: ignore[reportArgumentType]
	else:
		expectedWaveform = arrayTarget
	assert_allclose(actualWaveform, expectedWaveform, rtol, atol, 'stft', actual, lengthWaveform=arrayTarget.shape[-1], indexingAxis=indexingAxis, **keywordArguments)

@pytest.mark.parametrize('keywordArguments', [pytest.param({}, id='keywordArguments~None')])
@pytest.mark.parametrize('indexingAxis', [pytest.param(-1)])
@pytest.mark.parametrize('lengthWaveform', [pytest.param(0)])
@pytest.mark.parametrize('arrayTarget'
	, indirect=['arrayTarget']
	, argvalues=(
		pytest.param('waveform', id='arrayTarget~waveform')
		, pytest.param('arrayWaveforms', id='arrayTarget~arrayWaveforms')
	)
)
@pytest.mark.parametrize('pathFilename'
	, indirect=['pathFilename']
	, argvalues=(pytest.param('MusicNonVocal_bass_ch2_float32_s24_Hz48000_sec59.4_DC-.111.flac')
		,
	)
)
@pytest.mark.parametrize('listPathFilenames'
	, indirect=['listPathFilenames']
	, argvalues=(pytest.param(('MusicOnlyVocal_ch2_Hz44100_float32_sec20.1.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.2.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.4.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.15.wav'
			))
		,
	)
)
def test_Z0Z_stft(arrayTarget: Waveform | ArrayWaveforms, lengthWaveform: int, indexingAxis: int, keywordArguments: Parameters_stft, pathFilename: Path, listPathFilenames: tuple[Path, ...], rtol: float, atol: float) -> None:
	"""Until I implement better tests, test the round-trip of `stft` and `istft`.

	Until I have fixtures that can start this test with `Spectrogram` or `ArraySpectrograms`, I will
	just start with `Waveform` or `ArrayWaveforms`.
	"""
	Z0Z_spectrograms: Spectrogram | ArraySpectrograms = stft(arrayTarget, lengthWaveform=lengthWaveform, indexingAxis=indexingAxis, **keywordArguments)

	lengthWaveformTarget: int = arrayTarget.shape[getAxis()['time'].number]

	Z0Z_waveforms: Waveform | ArrayWaveforms = stft(Z0Z_spectrograms, lengthWaveform=lengthWaveformTarget, indexingAxis=indexingAxis, **keywordArguments)

	if numpy.issubdtype(arrayTarget.dtype, numpy.integer):
		expectedWaveform = amplitudeIntegerToFloating(arrayTarget.copy())  # pyright: ignore[reportArgumentType]
	else:
		expectedWaveform = arrayTarget

	assert_allclose(Z0Z_waveforms, expectedWaveform, rtol, atol, 'stft', Z0Z_spectrograms, lengthWaveform=lengthWaveformTarget, indexingAxis=indexingAxis, **keywordArguments)
