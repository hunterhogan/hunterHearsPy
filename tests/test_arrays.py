from __future__ import annotations

from hunterHearsPy import loadWaveforms, OptionsAlign, Parameters_loadWaveforms
from tests import assert_array_equal
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from collections.abc import Sequence
	from hunterHearsPy import ArrayWaveforms
	from pathlib import Path

@pytest.mark.parametrize(('listPathFilenames', 'keywordArguments')
	, indirect=['listPathFilenames']
	, argvalues=(
		pytest.param(('MusicOnlyVocal_ch2_Hz44100_float32_sec20.1.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.2.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.4.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.15.wav'
			), {}
			, id='MusicOnlyVocal'
		)
		, pytest.param(('MusicOnlyVocal_ch2_Hz44100_float32_sec20.1.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.2.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.4.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.15.wav'
			), Parameters_loadWaveforms(align='start', dtype='float32', dtype_str='float32', sampleRateDesired=44100)
			, id='MusicOnlyVocal__Parameters_loadWaveforms~defaults'
		)
		, pytest.param(('MusicNonVocal_bass_ch2_float32_s24_Hz48000_sec59.6.flac'
				, 'MusicNonVocal_bass_ch2_float32_s24_Hz48000_sec59.4_DC-.111.flac'
				, 'MusicNonVocal_bass_ch2_float32_s24_Hz48000_sec60.flac'
				, 'MusicNonVocal_bass_ch2_float32_s24_Hz48000_sec59.8.flac'
			), Parameters_loadWaveforms(align='center', sampleRateDesired=48000)
			, id='MusicNonVocal_bass'
		)
	)
)
def test_loadWaveforms(listPathFilenames: Sequence[Path], CPUlimit: int, keywordArguments: Parameters_loadWaveforms, expected: ArrayWaveforms) -> None:
	actual: ArrayWaveforms = loadWaveforms(listPathFilenames, CPUlimit=CPUlimit, **keywordArguments)

	assert_array_equal(actual, expected, 'loadWaveforms', listPathFilenames, CPUlimit=CPUlimit, **keywordArguments)





@pytest.mark.parametrize(('listPathFilenames', 'keywordArguments', 'align')
	, indirect=['listPathFilenames']
	, argvalues=(
		pytest.param(('MusicOnlyVocal_ch2_Hz44100_float32_sec20.1.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.2.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.4.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.15.wav'
			)
			, {}
			, 'start'
			, id='MusicOnlyVocal'
		)
		, pytest.param(('MusicOnlyVocal_ch2_Hz44100_float32_sec20.1.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.2.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.4.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.15.wav'
			)
			, Parameters_loadWaveforms(align='start', dtype='float32', dtype_str='float32', sampleRateDesired=44100)
			, 'start'
			, id='MusicOnlyVocal__Parameters_loadWaveforms~defaults'
		)
	)
)
def Z0Z_test_loadWaveforms_align(listPathFilenames: Sequence[Path], CPUlimit: int, keywordArguments: Parameters_loadWaveforms, align: OptionsAlign) -> None:
	pass
