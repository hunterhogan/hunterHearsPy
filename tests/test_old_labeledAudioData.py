from __future__ import annotations

from hunterHearsPy import readAudioFile
from tests.conftest import ingestSampleData, sampleData, WaveformAndMetadata
from typing import Final
import numpy
import pytest

# Constants for test validation
TOLERANCE_LUFS: Final[float] = 3.0  # LUFS measurement tolerance
LUFS_LOWER_BOUND: Final[float] = -100.0  # Reasonable lower bound for LUFS
def sampleData44100() -> list[WaveformAndMetadata]:
	return [dataSample for dataSample in ingestSampleData() if dataSample.sampleRate == 44100]

def sampleData48000() -> list[WaveformAndMetadata]:
	return [dataSample for dataSample in ingestSampleData() if dataSample.sampleRate == 48000]

@pytest.mark.parametrize('waveformData', sampleData44100())
def test_readLabeledAudioFiles44100(waveformData: WaveformAndMetadata) -> None:
	"""Test reading all labeled 44.1kHz audio files."""
	waveformResult = readAudioFile(waveformData.pathFilename)

	# Verify basic properties
	assert waveformResult.dtype.name == 'float32'

	# Verify waveform has reasonable audio content
	assert waveformResult.shape[1] > 0  # Has samples
	amplitudeMax = abs(waveformResult).max()
	assert amplitudeMax > 0.0  # Not silent
	assert amplitudeMax <= 1.0  # Not clipped

@pytest.mark.parametrize('waveformData', sampleData48000())
def test_readLabeledAudioFiles48000(waveformData: WaveformAndMetadata) -> None:
	"""Test reading all labeled 48kHz audio files."""
	waveformResult = readAudioFile(waveformData.pathFilename)

	# Verify basic properties
	assert waveformResult.dtype.name == 'float32'

	# Verify waveform has reasonable audio content
	assert waveformResult.shape[1] > 0  # Has samples
	amplitudeMax = abs(waveformResult).max()
	assert amplitudeMax > 0.0  # Not silent
	assert amplitudeMax <= 1.0  # Not clipped

def test_consistentResultsAcrossReads() -> None:
	"""Test that reading the same file multiple times gives identical results."""
	# Use the first available sample
	allSamples = sampleData()
	if not allSamples:
		pytest.skip('No sample data available')

	waveformData = allSamples[0]

	# Read the same file multiple times
	waveformFirst = readAudioFile(waveformData.pathFilename)
	waveformSecond = readAudioFile(waveformData.pathFilename)
	waveformThird = readAudioFile(waveformData.pathFilename)

	# Results should be identical
	numpy.testing.assert_array_equal(waveformFirst, waveformSecond)
	numpy.testing.assert_array_equal(waveformSecond, waveformThird)

@pytest.mark.parametrize('waveformData', sampleData()[:5])  # Test first 5 files only for performance
def test_metadataAccuracy(waveformData: WaveformAndMetadata) -> None:
	"""Test that file metadata matches expected values from filename parsing."""
	# Test that the sample rate information is reasonable
	# We can't directly test sample rate from the waveform alone,
	# but we can test that the parsed metadata is sensible
	assert waveformData.sampleRate in {44100, 48000}
	assert waveformData.channelsTotal in {1, 2}
	assert waveformData.LUFS < 0  # LUFS should be negative (below digital full scale)
	assert waveformData.LUFS > LUFS_LOWER_BOUND

	# Test ID field has meaningful content
	assert len(waveformData.ID) > 0
	assert waveformData.ID != 'unknown'
