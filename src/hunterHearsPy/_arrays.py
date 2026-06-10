# pyright: reportUnnecessaryComparison=false
# pyright: reportUnusedVariable=false
# ruff: noqa: F841 ERA001
from __future__ import annotations

from hunterHearsPy import parametersUniversal, universalDtypeSpectrogram, universalDtypeWaveform
from hunterHearsPy._fft import stft
from hunterHearsPy._io import readAudioFile
from hunterHearsPy.theTypes import WaveformMetadata
from multiprocessing import set_start_method as multiprocessing_set_start_method
from tqdm.auto import tqdm
from typing import cast, TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from collections.abc import Sequence
	from hunterHearsPy.theTypes import ArraySpectrograms, ArrayWaveforms, Spectrogram, Waveform
	from os import PathLike
	from typing import Any

if __name__ == '__main__':
	multiprocessing_set_start_method('spawn')

def getWaveformMetadata(listPathFilenames: Sequence[str | PathLike[str]], sampleRate: float) -> dict[int, WaveformMetadata]:
	"""Retrieve metadata for a collection of audio waveform files.

	You can use this function to inspect the length of each audio file before loading
	waveforms into memory. `getWaveformMetadata` reads each file at `sampleRate`, measures
	the sample count, and returns one `WaveformMetadata` [1] record per file indexed by
	position in `listPathFilenames`. Each record's `samplesLeading` and `samplesTrailing`
	fields are initialized to `0`; callers may adjust them before passing the result to
	downstream loaders such as `loadWaveforms` or `loadSpectrograms`.

	Parameters
	----------
	listPathFilenames : Sequence[str | PathLike[str]]
		Sequence of paths to audio files.
	sampleRate : float
		Target sample rate used when reading each file to measure its length in samples.

	Returns
	-------
	dictionaryWaveformMetadata : dict[int, WaveformMetadata]
		Dictionary mapping each integer index to a `WaveformMetadata` [1] record. Each
		record contains `pathFilename` (string path), `lengthWaveform` (sample count at
		`sampleRate`), `samplesLeading` (initialized to `0`), and `samplesTrailing`
		(initialized to `0`).

	File Reading Progress
	---------------------
	`tqdm` [2] displays a progress bar in the terminal while `getWaveformMetadata` reads
	each file in `listPathFilenames`.

	References
	----------
	[1] `WaveformMetadata`

	[2] tqdm — fast, extensible progress bar for Python and CLI
		https://tqdm.github.io/

	"""
	axisTime: int = -1
	dictionaryWaveformMetadata: dict[int, WaveformMetadata] = {}
	for index, pathFilename in enumerate(tqdm(listPathFilenames)):
		lengthWaveform = readAudioFile(pathFilename, sampleRate).shape[axisTime]
		dictionaryWaveformMetadata[index] = WaveformMetadata(
			pathFilename=str(pathFilename),
			lengthWaveform=lengthWaveform,
			samplesLeading=0,
			samplesTrailing=0,
		)
	return dictionaryWaveformMetadata

def loadWaveforms(listPathFilenames: Sequence[str | PathLike[str]], sampleRateTarget: float | None = None) -> ArrayWaveforms:
	"""Load a list of audio files into a single stacked NumPy array.

	You can use this function to batch-load multiple audio files into one `ArrayWaveforms` [1]
	array. All waveforms are resampled to `sampleRateTarget`, converted to stereo when
	necessary, and zero-padded on the trailing edge to match the length of the longest
	waveform. The resulting array is shaped `(channels, lengthWaveformMaximum, countFiles)`.

	Parameters
	----------
	listPathFilenames : Sequence[str | PathLike[str]]
		List of paths to audio files.
	sampleRateTarget : float | None = None
		Target sample rate in Hz. Defaults to `44100` when `None`.

	Returns
	-------
	arrayWaveforms : ArrayWaveforms
		Stacked waveform data shaped `(2, lengthWaveformMaximum, countFiles)` as `float32`,
		where `lengthWaveformMaximum` is the maximum sample count across all files at
		`sampleRateTarget`.

	Zero-Padding
	------------
	Waveforms shorter than `lengthWaveformMaximum` are zero-padded on the trailing edge.
	Leading padding is applied when `WaveformMetadata.samplesLeading` [2] is non-zero;
	`getWaveformMetadata` initializes `samplesLeading` to `0` by default.

	References
	----------
	[1] `ArrayWaveforms`

	[2] `WaveformMetadata`
	"""
	if sampleRateTarget is None:
		sampleRateTarget = parametersUniversal['sampleRate']

	axisOrderMapping: dict[str, int] = {'indexingAxis': -1, 'axisTime': -2, 'axisChannels': 0}
	axesSizes: dict[str, int] = dict.fromkeys(axisOrderMapping.keys(), 1)
	countAxes: int = len(axisOrderMapping)
	listShapeIndexToSize: list[int] = [9001] * countAxes

	countWaveforms: int = len(listPathFilenames)
	axesSizes['indexingAxis'] = countWaveforms
	countChannels: int = 2
	axesSizes['axisChannels'] = countChannels

	axisTime: int = -1
	dictionaryWaveformMetadata: dict[int, WaveformMetadata] = getWaveformMetadata(listPathFilenames, sampleRateTarget)
	samplesTotalMaximum = max(entry['lengthWaveform'] + entry['samplesLeading'] + entry['samplesTrailing'] for entry in dictionaryWaveformMetadata.values())
	axesSizes['axisTime'] = samplesTotalMaximum

	for keyName, axisSize in axesSizes.items():
		axisNormalized: int = (axisOrderMapping[keyName] + countAxes) % countAxes
		listShapeIndexToSize[axisNormalized] = axisSize
	tupleShapeArray: tuple[int, int, int] = cast('tuple[int, int, int]', tuple(listShapeIndexToSize))

	arrayWaveforms: ArrayWaveforms = numpy.zeros(tupleShapeArray, dtype=universalDtypeWaveform)

	for index, metadata in dictionaryWaveformMetadata.items():
		waveform: Waveform = readAudioFile(metadata['pathFilename'], sampleRateTarget)
		samplesTrailing = metadata['lengthWaveform'] + metadata['samplesLeading'] - samplesTotalMaximum
		if samplesTrailing == 0:
			samplesTrailing = None
		# TODO Add padding logic to `loadWaveforms` and `loadSpectrograms`
		arrayWaveforms[:, metadata['samplesLeading']:samplesTrailing, index] = waveform

	return arrayWaveforms

def _getSpectrogram(waveform: Waveform, metadata: WaveformMetadata, sampleRateTarget: float, **parametersSTFT: Any) -> Spectrogram:
	"""I use this to load a single audio file into a pre-allocated waveform buffer and compute its spectrogram.

	(AI generated docstring)

	I use this shared subroutine inside `loadSpectrograms` to avoid reallocating a waveform
	buffer for each file. `_getSpectrogram` copies audio data from `metadata['pathFilename']`
	into the caller-provided `waveform` buffer at the position described by `metadata`, then
	computes `stft` with `sampleRateTarget` and any additional `parametersSTFT`. The caller
	must pass a fresh copy of the buffer for each iteration.

	Parameters
	----------
	waveform : Waveform
		Pre-allocated buffer into which audio data is copied before the STFT. The caller
		must pass a separate copy for each file to prevent data from accumulating across
		iterations.
	metadata : WaveformMetadata
		Record describing `pathFilename`, `lengthWaveform`, `samplesLeading`, and
		`samplesTrailing` for the audio file being loaded.
	sampleRateTarget : float
		Target sample rate passed to `readAudioFile`.
	**parametersSTFT : Any
		Keyword parameters forwarded to `stft`.

	Returns
	-------
	spectrogram : Spectrogram
		Complex spectrogram of `waveform` after copying the audio file into the buffer.

	"""
	# All waveforms have the same shape so that all spectrograms have the same shape.
	# TODO Add padding logic to `loadWaveforms` and `loadSpectrograms`
	lengthWaveform = metadata['lengthWaveform'] + metadata['samplesLeading'] + metadata['samplesTrailing']
	# All shorter waveforms are forced to have trailing zeros.
	waveform[:, 0:lengthWaveform] = readAudioFile(metadata['pathFilename'], sampleRateTarget)
	return stft(waveform, sampleRate=sampleRateTarget, **parametersSTFT)

def loadSpectrograms(listPathFilenames: Sequence[str | PathLike[str]], sampleRateTarget: float | None = None, **parametersSTFT: Any) -> tuple[ArraySpectrograms, dict[int, WaveformMetadata]]:
	"""Load spectrograms from a list of audio files.

	You can use this function to batch-convert audio files to spectrograms in a single call.
	`loadSpectrograms` reads each file, pads all waveforms to the same length, computes the
	Short-Time Fourier Transform for each, and stacks the results into one
	`ArraySpectrograms` [1] array. The function also returns a `WaveformMetadata` [2]
	dictionary that describes each file's original length and padding.

	Parameters
	----------
	listPathFilenames : Sequence[str | PathLike[str]]
		List of paths to audio files.
	sampleRateTarget : float | None = None
		Target sample rate in Hz. Defaults to `44100` when `None`.
	**parametersSTFT : Any
		Keyword parameters forwarded to `stft`, such as `lengthWindowingFunction` and
		`lengthHop`.

	Returns
	-------
	tupleSpectrogramsMetadata : tuple[ArraySpectrograms, dict[int, WaveformMetadata]]
		A two-element `tuple`. The first element is `ArraySpectrograms` [1] shaped
		`(channels, frequencies, frames, countFiles)` as `complex64`. The second element
		is a `dict` mapping integer file indices to `WaveformMetadata` [2] records.

	File Reading Progress
	---------------------
	`tqdm` [3] displays a progress bar in the terminal during the spectrogram computation
	loop.

	References
	----------
	[1] `ArraySpectrograms`

	[2] `WaveformMetadata`

	[3] tqdm — fast, extensible progress bar for Python and CLI
		https://tqdm.github.io/

	"""
	if sampleRateTarget is None:
		sampleRateTarget = parametersUniversal['sampleRate']

	max_workersHARDCODED: int = 3
	max_workers = max_workersHARDCODED

	dictionaryWaveformMetadata: dict[int, WaveformMetadata] = getWaveformMetadata(listPathFilenames, sampleRateTarget)

	samplesTotalMaximum: int = max(entry['lengthWaveform'] + entry['samplesLeading'] + entry['samplesTrailing'] for entry in dictionaryWaveformMetadata.values())
	countChannels = 2
	waveformTemplate: Waveform = numpy.zeros(shape=(countChannels, samplesTotalMaximum), dtype=universalDtypeWaveform)
	spectrogramTemplate: Spectrogram = stft(waveformTemplate, sampleRate=sampleRateTarget, **parametersSTFT)

	arraySpectrograms: ArraySpectrograms = numpy.zeros(shape=(*spectrogramTemplate.shape, len(dictionaryWaveformMetadata)), dtype=universalDtypeSpectrogram)

	for index, metadata in tqdm(dictionaryWaveformMetadata.items()):
		arraySpectrograms[..., index] = _getSpectrogram(waveformTemplate.copy(), metadata, sampleRateTarget, **parametersSTFT)

	# with ProcessPoolExecutor(max_workers) as concurrencyManager:
	# 	dictionaryConcurrency = {concurrencyManager.submit( # 		_getSpectrogram, waveformTemplate.copy(), metadata, sampleRateTarget, **parametersSTFT): index
	# 			for index, metadata in dictionaryWaveformMetadata.items()}

	# 	for claimTicket in tqdm(as_completed(dictionaryConcurrency), total=len(dictionaryConcurrency)):
	# 		arraySpectrograms[..., dictionaryConcurrency[claimTicket]] = claimTicket.result()

	return arraySpectrograms, dictionaryWaveformMetadata
