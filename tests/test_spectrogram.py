"""test_waveform or test_spectrogram? if a spectrogram is involved at any point, then test_spectrogram."""

from __future__ import annotations

from hunterHearsPy import loadSpectrograms, readAudioFile, stft, waveformSpectrogramWaveform
from tests.conftest import prototype_numpyAllClose, standardizedEqualTo, uniformTestFailureMessage, WaveformAndMetadata
from typing import Any, Final, TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from pathlib import Path

expectedSpectrogramDimensions: Final[int] = 4

@pytest.mark.parametrize('sampleRateDesired', [22050, 44100, 48000])
def test_loadSpectrograms_acceptssampleRateDesired(listPathFilenamesFromWaveformData: list[Path], sampleRateDesired: int) -> None:
	"""Test that loadSpectrograms accepts different target sample rates and produces valid spectrograms."""
	arraySpectrograms, dictionaryWaveformMetadata = loadSpectrograms(listPathFilenamesFromWaveformData, sampleRateDesired)

	expectedCountFiles = len(listPathFilenamesFromWaveformData)
	actualShape = arraySpectrograms.shape
	actualCountFiles = actualShape[-1]
	actualCountMetadata = len(dictionaryWaveformMetadata)

	assert actualCountFiles == expectedCountFiles, uniformTestFailureMessage(
		expectedCountFiles, actualCountFiles, 'loadSpectrograms', listPathFilenamesFromWaveformData, sampleRateDesired
	)
	assert actualCountMetadata == expectedCountFiles, uniformTestFailureMessage(
		expectedCountFiles, actualCountMetadata, 'loadSpectrograms metadata count', listPathFilenamesFromWaveformData, sampleRateDesired
	)
	assert len(actualShape) == expectedSpectrogramDimensions, uniformTestFailureMessage(
		expectedSpectrogramDimensions
		, len(actualShape)
		, 'loadSpectrograms shape dimensions'
		, listPathFilenamesFromWaveformData
		, sampleRateDesired
	)
	assert numpy.issubdtype(arraySpectrograms.dtype, numpy.complexfloating), uniformTestFailureMessage(
		'complex floating point type'
		, arraySpectrograms.dtype
		, 'loadSpectrograms dtype'
		, listPathFilenamesFromWaveformData
		, sampleRateDesired
	)

@pytest.mark.parametrize('lengthWindowingFunctionSTFT,lengthHopSTFT', [(1024, 256), (2048, 512), (4096, 1024)])
def test_loadSpectrograms_acceptsSTFTParameters(
	listPathFilenamesFromWaveformData: list[Path], lengthWindowingFunctionSTFT: int, lengthHopSTFT: int
) -> None:
	"""Test that loadSpectrograms accepts custom STFT parameters and produces spectrograms with expected shapes."""
	sampleRateDesired = 44100

	arraySpectrograms, _dictionaryWaveformMetadata = loadSpectrograms(
		listPathFilenamesFromWaveformData
		, sampleRateDesired
		, lengthWindowingFunction=lengthWindowingFunctionSTFT
		, lengthHop=lengthHopSTFT
	)

	waveformSingle = readAudioFile(listPathFilenamesFromWaveformData[0], sampleRateDesired)
	spectrogramExpected = stft(
		waveformSingle, sampleRate=sampleRateDesired, lengthWindowingFunction=lengthWindowingFunctionSTFT, lengthHop=lengthHopSTFT
	)

	expectedShape = spectrogramExpected.shape
	actualShape = arraySpectrograms.shape[:-1]

	assert actualShape == expectedShape, uniformTestFailureMessage(
		expectedShape
		, actualShape
		, 'loadSpectrograms STFT shape'
		, listPathFilenamesFromWaveformData
		, sampleRateDesired
		, lengthWindowingFunction=lengthWindowingFunctionSTFT
		, lengthHop=lengthHopSTFT
	)

def test_loadSpectrograms_singleFile(waveformDataStereo44kHz: WaveformAndMetadata) -> None:
	"""Test loading a spectrogram from a single file."""
	sampleRateDesired = 44100
	listPathFilenameSingle = [waveformDataStereo44kHz.pathFilename]

	arraySpectrograms, dictionaryWaveformMetadata = loadSpectrograms(listPathFilenameSingle, sampleRateDesired)

	waveformSingle = readAudioFile(waveformDataStereo44kHz.pathFilename, sampleRateDesired)
	spectrogramExpected = stft(waveformSingle, sampleRate=sampleRateDesired)

	expectedShape = spectrogramExpected.shape
	actualShape = arraySpectrograms.shape[:-1]
	expectedCountFiles = 1
	actualCountFiles = arraySpectrograms.shape[-1]
	expectedCountMetadata = 1
	actualCountMetadata = len(dictionaryWaveformMetadata)

	assert actualShape == expectedShape, uniformTestFailureMessage(
		expectedShape, actualShape, 'loadSpectrograms single file shape', listPathFilenameSingle, sampleRateDesired
	)
	assert actualCountFiles == expectedCountFiles, uniformTestFailureMessage(
		expectedCountFiles, actualCountFiles, 'loadSpectrograms single file count', listPathFilenameSingle, sampleRateDesired
	)
	assert actualCountMetadata == expectedCountMetadata, uniformTestFailureMessage(
		expectedCountMetadata
		, actualCountMetadata
		, 'loadSpectrograms single file metadata count'
		, listPathFilenameSingle
		, sampleRateDesired
	)

def test_loadSpectrograms_roundTripReconstructionAccuracy(waveformDataStereo44kHz: WaveformAndMetadata) -> None:
	"""Test that loadSpectrograms produces spectrograms that roundtrip through direct STFT operations."""
	sampleRateDesired = 44100
	listPathFilenameSingle = [waveformDataStereo44kHz.pathFilename]

	arraySpectrograms, _dictionaryWaveformMetadata = loadSpectrograms(listPathFilenameSingle, sampleRateDesired)

	waveformOriginal = readAudioFile(waveformDataStereo44kHz.pathFilename, sampleRateDesired)
	spectrogramDirect = stft(waveformOriginal, sampleRate=sampleRateDesired)

	expectedShape = spectrogramDirect.shape
	actualShape = arraySpectrograms.shape[:-1]
	expectedCountFiles = 1
	actualCountFiles = arraySpectrograms.shape[-1]

	assert actualShape == expectedShape, uniformTestFailureMessage(
		expectedShape, actualShape, 'loadSpectrograms roundtrip shape comparison', listPathFilenameSingle, sampleRateDesired
	)
	assert actualCountFiles == expectedCountFiles, uniformTestFailureMessage(
		expectedCountFiles, actualCountFiles, 'loadSpectrograms roundtrip file count', listPathFilenameSingle, sampleRateDesired
	)

def test_loadSpectrograms_rejectsEmptyInput() -> None:
	"""Test that loadSpectrograms raises ValueError for empty input."""
	standardizedEqualTo(ValueError, loadSpectrograms, [], 44100)

def test_stft_forwardTransform(waveformDataStereo44kHz: WaveformAndMetadata) -> None:
	"""Test that stft produces complex-valued spectrograms from real waveforms."""
	waveformSingle = waveformDataStereo44kHz.waveform

	spectrogram = stft(waveformSingle)

	actualDtype = spectrogram.dtype
	expectedComplexFloating = True
	actualComplexFloating = numpy.issubdtype(actualDtype, numpy.complexfloating)
	expectedNonEmpty = True
	actualNonEmpty = spectrogram.shape[0] > 0 and spectrogram.shape[1] > 0

	assert actualComplexFloating == expectedComplexFloating, uniformTestFailureMessage(
		'complex floating point type', actualDtype, 'stft forward transform dtype', waveformSingle
	)
	assert actualNonEmpty == expectedNonEmpty, uniformTestFailureMessage(
		'non-empty spectrogram', spectrogram.shape, 'stft forward transform shape', waveformSingle
	)

def test_stft_inverseTransform(waveformDataStereo44kHz: WaveformAndMetadata) -> None:
	"""Test that stft inverse transform reconstructs waveforms accurately."""
	waveformOriginal = waveformDataStereo44kHz.waveform

	spectrogram = stft(waveformOriginal)
	waveformReconstructed = stft(spectrogram, inverse=True, lengthWaveform=waveformOriginal.shape[1])

	prototype_numpyAllClose(waveformOriginal, 1e-2, 1e-2, lambda: waveformReconstructed)

@pytest.mark.parametrize('lengthWindowingFunctionSTFT,lengthHopSTFT', [(1024, 256), (2048, 512), (4096, 1024)])
def test_stft_acceptsSTFTParameters(
	waveformDataStereo44kHz: WaveformAndMetadata, lengthWindowingFunctionSTFT: int, lengthHopSTFT: int
) -> None:
	"""Test that stft accepts different windowing and hop parameters."""
	waveformSingle = waveformDataStereo44kHz.waveform
	sampleRate = waveformDataStereo44kHz.sampleRate

	spectrogram = stft(
		waveformSingle, sampleRate=sampleRate, lengthWindowingFunction=lengthWindowingFunctionSTFT, lengthHop=lengthHopSTFT
	)

	expectedNonEmpty = True
	actualNonEmpty = spectrogram.shape[0] > 0 and spectrogram.shape[1] > 0
	expectedComplexFloating = True
	actualComplexFloating = numpy.issubdtype(spectrogram.dtype, numpy.complexfloating)

	assert actualNonEmpty == expectedNonEmpty, uniformTestFailureMessage(
		'non-empty spectrogram'
		, spectrogram.shape
		, 'stft with custom parameters shape'
		, waveformSingle
		, sampleRate=sampleRate
		, lengthWindowingFunction=lengthWindowingFunctionSTFT
		, lengthHop=lengthHopSTFT
	)
	assert actualComplexFloating == expectedComplexFloating, uniformTestFailureMessage(
		'complex floating point type'
		, spectrogram.dtype
		, 'stft with custom parameters dtype'
		, waveformSingle
		, sampleRate=sampleRate
		, lengthWindowingFunction=lengthWindowingFunctionSTFT
		, lengthHop=lengthHopSTFT
	)

def test_stft_rejectsInverseWithoutLengthWaveform(waveformDataStereo44kHz: WaveformAndMetadata) -> None:
	"""Test that stft raises ValueError when inverse=True but lengthWaveform is not provided."""
	waveformSingle = waveformDataStereo44kHz.waveform
	spectrogram = stft(waveformSingle)

	standardizedEqualTo(ValueError, stft, spectrogram, inverse=True)

def test_waveformSpectrogramWaveform_identityTransform(waveformDataStereo44kHz: WaveformAndMetadata) -> None:
	"""Test that waveformSpectrogramWaveform with identity function preserves waveforms."""
	waveformOriginal = waveformDataStereo44kHz.waveform

	def identitySpectrogram(spectrogram: Any) -> Any:
		return spectrogram

	processor = waveformSpectrogramWaveform(identitySpectrogram)
	waveformProcessed = processor(waveformOriginal)

	prototype_numpyAllClose(waveformOriginal, 1e-2, 1e-2, lambda: waveformProcessed)

def test_waveformSpectrogramWaveform_magnitudeOnlyTransform(waveformDataStereo44kHz: WaveformAndMetadata) -> None:
	"""Test that waveformSpectrogramWaveform with magnitude-only processing changes the waveform."""
	waveformOriginal = waveformDataStereo44kHz.waveform

	def magnitudeOnlySpectrogram(spectrogram: Any) -> Any:
		return numpy.abs(spectrogram)

	processor = waveformSpectrogramWaveform(magnitudeOnlySpectrogram)
	waveformProcessed = processor(waveformOriginal)

	expectedShapeMatch = True
	actualShapeMatch = waveformOriginal.shape == waveformProcessed.shape
	expectedContentDifferent = True
	actualContentDifferent = not numpy.allclose(waveformOriginal, waveformProcessed, atol=1e-2, rtol=1e-2)

	assert actualShapeMatch == expectedShapeMatch, uniformTestFailureMessage(
		waveformOriginal.shape
		, waveformProcessed.shape
		, 'waveformSpectrogramWaveform magnitude-only shape preservation'
		, waveformOriginal
	)
	assert actualContentDifferent == expectedContentDifferent, uniformTestFailureMessage(
		'different waveform content'
		, 'nearly identical content'
		, 'waveformSpectrogramWaveform magnitude-only content change'
		, waveformOriginal
	)
