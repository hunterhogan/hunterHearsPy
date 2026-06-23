from __future__ import annotations

from hunterHearsPy import loadWaveforms, Parameters_loadWaveforms
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
		pytest.param(['MusicOnlyVocal_ch2_Hz44100_float32_sec20.1.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.2.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.4.wav'
				, 'MusicOnlyVocal_ch2_Hz44100_float32_sec20.15.wav'
			], Parameters_loadWaveforms({})
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
	)
)
def test_loadWaveforms(listPathFilenames: Sequence[Path], CPUlimit: int, keywordArguments: Parameters_loadWaveforms, expected: ArrayWaveforms) -> None:
	actual: ArrayWaveforms = loadWaveforms(listPathFilenames, CPUlimit=CPUlimit, **keywordArguments)

	assert_array_equal(actual, expected, 'loadWaveforms', listPathFilenames, CPUlimit=CPUlimit, **keywordArguments)
