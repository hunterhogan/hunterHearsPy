from __future__ import annotations

from hunterHearsPy import ArrayWaveforms, loadWaveforms, normalizeArrayWaveforms, normalizeWaveform, Waveform
from hunterHearsPy.amplitude import amplitudeToSoundfile
from tests.conftest import pathDataSamples_labeled, sampleData
from typing import Final, TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from pathlib import Path

rtolDEFAULT: Final[float] = 1e-7
amplitudeNorm: Final[float] = 1.0

listFilenamesSameShape = [
	'WAV_44100_ch2_sec5_Sine_Copy0.wav',
	'WAV_44100_ch2_sec5_Sine_Copy1.wav',
	'WAV_44100_ch2_sec5_Sine_Copy2.wav',
	'WAV_44100_ch2_sec5_Sine_Copy3.wav',
]

@pytest.mark.parametrize(
	'arrayTarget, expected',
	[
		pytest.param(
			numpy.array([[-233, 89, 144]], dtype=numpy.int16),
			numpy.array([[-233, 89, 144]], dtype=numpy.int16),
			id='int16Passthrough',
		),
		pytest.param(
			numpy.array([[-128, -34, 55, 127]], dtype=numpy.int8),
			numpy.array([[-32768, -8704, 14080, 32512]], dtype=numpy.int16),
			id='int8ScaledToInt16',
		),
		pytest.param(
			numpy.array([[0, 233, 32768, 65535]], dtype=numpy.uint16),
			numpy.array([[-32768, -32535, 0, 32767]], dtype=numpy.int16),
			id='uint16CenteredToInt16',
		),
		pytest.param(
			numpy.array([[-0.5, 0.125, 0.75]], dtype=numpy.float16),
			numpy.array([[-0.5, 0.125, 0.75]], dtype=numpy.float32),
			id='float16CastToFloat32',
		),
	],
)
def test_amplitudeToSoundfile(arrayTarget: Waveform, expected: Waveform) -> None:
	actual: Waveform = amplitudeToSoundfile(arrayTarget)

	assert actual.dtype == expected.dtype, (
		f'amplitudeToSoundfile returned dtype `{actual.dtype}` instead of `{expected.dtype}` '
		f'for input dtype `{arrayTarget.dtype}`.'
	)
	numpy.testing.assert_array_equal(
		actual,
		expected,
		err_msg=f'amplitudeToSoundfile returned unexpected values for input dtype `{arrayTarget.dtype}`.',
	)

@pytest.fixture
def listPathFilenamesArrayWaveforms() -> list[Path]:
	return [pathDataSamples_labeled / filename for filename in listFilenamesSameShape]

@pytest.fixture
def array44100_ch2_sec5_Sine(listPathFilenamesArrayWaveforms: list[Path]) -> ArrayWaveforms:
	"""
	Load the four WAV files with the same shape into an array.

	Returns:
		arrayWaveforms: Array of waveforms with shape (channels, samples, count_of_waveforms)
	"""
	return loadWaveforms(listPathFilenamesArrayWaveforms)

@pytest.mark.parametrize(
	'ID, waveform, sampleRate, LUFS, channelsTotal',
	[(dataSample.ID, dataSample.waveform, dataSample.sampleRate, dataSample.LUFS, dataSample.channelsTotal) for dataSample in sampleData()],
)
def test_normalize_peak_amplitude(ID: str, waveform: Waveform, sampleRate: float, LUFS: float, channelsTotal: int) -> None:
	"""Test that normalize() scales waveform to have peak amplitude equal to amplitudeNorm."""
	waveformNormalized, _DISCARDrevertFunction = normalizeWaveform(waveform)

	peakAbsolute = numpy.max(numpy.abs(waveformNormalized))
	assert numpy.isclose(peakAbsolute, amplitudeNorm, rtol=1e-5), f'Peak amplitude {peakAbsolute} should equal {amplitudeNorm} for {ID}'

@pytest.mark.parametrize(
	'ID, waveform, sampleRate, LUFS, channelsTotal',
	[(dataSample.ID, dataSample.waveform, dataSample.sampleRate, dataSample.LUFS, dataSample.channelsTotal) for dataSample in sampleData()],
)
def test_normalize_reversion(ID: str, waveform: Waveform, sampleRate: float, LUFS: float, channelsTotal: int) -> None:
	"""Test that normalize() returns a reversion function that restores the original waveform."""
	waveformNormalized, revertNormalization = normalizeWaveform(waveform.copy())

	waveformReverted = revertNormalization(waveformNormalized)

	assert numpy.allclose(waveformReverted, waveform, rtol=1e-5), f'Reverted waveform should match original for {ID}'

@pytest.mark.parametrize(
	'ID, waveform, sampleRate, LUFS, channelsTotal',
	[(dataSample.ID, dataSample.waveform, dataSample.sampleRate, dataSample.LUFS, dataSample.channelsTotal) for dataSample in sampleData()],
)
def test_normalize_preserves_relative_amplitudes(ID: str, waveform: Waveform, sampleRate: float, LUFS: float, channelsTotal: int) -> None:
	"""Test that normalize() preserves relative amplitudes between samples."""
	# Create reference points to compare
	indexReference1, indexReference2 = 1000, 2000
	if indexReference2 >= waveform.shape[1]:
		indexReference1, indexReference2 = 10, 20

	if waveform.shape[0] >= 2:
		# For stereo or multichannel
		ratioOriginal = waveform[0, indexReference1] / (waveform[1, indexReference2] + 1e-10)

		waveformNormalized, _DISCARDrevertFunction = normalizeWaveform(waveform.copy())
		ratioNormalized = waveformNormalized[0, indexReference1] / (waveformNormalized[1, indexReference2] + 1e-10)

		assert numpy.isclose(ratioOriginal, ratioNormalized, rtol=1e-5), f'Relative amplitudes should be preserved for {ID}'

def test_normalizeArrayWaveforms(array44100_ch2_sec5_Sine: ArrayWaveforms) -> None:
	"""Test that normalizeArrayWaveforms scales multiple waveforms to have peak amplitude equal to amplitudeNorm."""
	# Save a copy of the original array for comparison after reversion
	arrayOriginal = array44100_ch2_sec5_Sine.copy()

	# Apply normalization to all waveforms in the array
	arrayNormalized, listRevertNormalization = normalizeArrayWaveforms(array44100_ch2_sec5_Sine.copy())

	# Test 1: Check that each waveform is normalized to the correct peak amplitude
	for indexWaveform in range(arrayNormalized.shape[-1]):
		waveformCurrent = arrayNormalized[..., indexWaveform]
		peakAbsolute = numpy.max(numpy.abs(waveformCurrent))
		assert numpy.isclose(peakAbsolute, amplitudeNorm, rtol=rtolDEFAULT), (
			f'Peak amplitude {peakAbsolute} should equal {amplitudeNorm} for waveform at index {indexWaveform}'
		)

	# Test 2: Check that reversion functions restore original waveforms
	arrayReverted = arrayNormalized.copy()
	for indexWaveform in range(arrayReverted.shape[-1]):
		arrayReverted[..., indexWaveform] = listRevertNormalization[indexWaveform](arrayReverted[..., indexWaveform])

	assert numpy.allclose(arrayReverted, arrayOriginal, rtol=rtolDEFAULT), 'Reverted array should match the original array'
