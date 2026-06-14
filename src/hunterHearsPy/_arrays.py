# pyright: reportAssignmentType=false
# pyright: reportUnnecessaryComparison=false
# pyright: reportUnusedVariable=false
# ty:ignore[invalid-assignment]
from __future__ import annotations

from hunterHearsPy import FileDescriptorOrPath, getAxis, readAudioFile, setting, stft, WaveformAxes, WaveformMetadata
from math import ceil as ceiling
from tqdm.auto import tqdm
from typing import TYPE_CHECKING
import numpy
import sys

if TYPE_CHECKING:
	from collections.abc import Sequence
	from hunterHearsPy import ArraySpectrograms, ArrayWaveforms, Spectrogram, Waveform
	from typing import Any

# TODO concurrency
# TODO more sophisticated tests

def getWaveformMetadata(listPathFilenames: Sequence[FileDescriptorOrPath], sampleRate: float) -> tuple[dict[int, WaveformMetadata], dict[str, WaveformAxes]]:
	"""Retrieve metadata for a collection of audio waveform files.

	You can use this function to inspect the length of each audio file before loading
	waveforms into memory. `getWaveformMetadata` reads each file at `sampleRate`, measures
	the sample count, and returns one `WaveformMetadata` [1] record per file indexed by
	position in `listPathFilenames`. Each record's `samplesStart` and `samplesStop`
	fields are initialized to `0`; callers may adjust them before passing the result to
	downstream loaders such as `loadWaveforms` or `loadSpectrograms`.

	Parameters
	----------
	listPathFilenames : Sequence[FileDescriptorOrPath]
		Sequence of paths to audio files.
	sampleRate : float
		Target sample rate used when reading each file to measure its length in samples.

	Returns
	-------
	dictionaryWaveformMetadata : dict[int, WaveformMetadata]
		Dictionary mapping each integer index to a `WaveformMetadata` [1] record. Each
		record contains `pathFilename` (string path), `lengthWaveform` (sample count at
		`sampleRate`), `samplesStart` (initialized to `0`), and `samplesStop`
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
#======== Initialize ==========================================================

	axis: dict[str, WaveformAxes] = getAxis()
	channelMaximum: int = 0
	dictionaryWaveformMetadata: dict[int, WaveformMetadata] = {}
	lengthMaximum: int = 0

#======== Populate ===========================================================

	if len(listPathFilenames) == 0:
		message: str = (f"I received `{len(listPathFilenames) = }`, so `arrayWaveforms` will have zero-sized axes.")
		sys.stderr.write(message + '\n')

	for index, pathFilename in enumerate(tqdm(listPathFilenames, desc='Preparing combined array', leave=False)):
		channels, lengthWaveform = readAudioFile(pathFilename, sampleRate).shape
		dictionaryWaveformMetadata[index] = WaveformMetadata(
			channels=channels
			, lengthWaveform=lengthWaveform
			, pathFilename=pathFilename
			, samplesStart=0
			, samplesStop=0
		)
		channelMaximum = max(channelMaximum, channels)
		lengthMaximum = max(lengthMaximum, lengthWaveform)

	axis['channel'] = WaveformAxes(number=axis['channel'].number, size=channelMaximum)
	axis['time'] = WaveformAxes(number=axis['time'].number, size=lengthMaximum)
	axis['indexing'] = WaveformAxes(number=axis['indexing'].number, size=len(listPathFilenames))

#======== Calculate ===========================================================

	multiplicandSamplesStart: float = max((setting.align == 'center') / 2, setting.align == 'start')

	for metadata in dictionaryWaveformMetadata.values():
		samplesPadding: int = axis['time'].size - metadata['lengthWaveform']
		# TODO document that is `samplesPadding` is odd, the extra pad-sample is added to samplesStart.
		# TODO think about whether I want the extra pad-sample at the start or end.
		metadata['samplesStart'] = ceiling(samplesPadding * multiplicandSamplesStart)
		metadata['samplesStop'] = metadata['samplesStart'] + metadata['lengthWaveform']

	return dictionaryWaveformMetadata, axis

def loadWaveforms(listPathFilenames: Sequence[FileDescriptorOrPath], sampleRateDesired: float | None = None) -> ArrayWaveforms:
	"""Load a list of audio files into a single stacked NumPy array.

	You can use this function to batch-load multiple audio files into one `ArrayWaveforms` [1]
	array. All waveforms are resampled to `sampleRateDesired`, converted to stereo when
	necessary, and zero-padded on the trailing edge to match the length of the longest
	waveform. The resulting array is shaped `(channels, lengthWaveformMaximum, countFiles)`.

	Parameters
	----------
	listPathFilenames : Sequence[FileDescriptorOrPath]
		List of paths to audio files.
	sampleRateDesired : float | None = None
		Target sample rate in Hz. Defaults to `44100` when `None`.

	Returns
	-------
	arrayWaveforms : ArrayWaveforms
		Stacked waveform data shaped `(2, lengthWaveformMaximum, countFiles)` as `float32`,
		where `lengthWaveformMaximum` is the maximum sample count across all files at
		`sampleRateDesired`.

	Zero-Padding
	------------
	Waveforms shorter than `lengthWaveformMaximum` are zero-padded on the trailing edge.
	Leading padding is applied when `WaveformMetadata.samplesStart` [2] is non-zero;
	`getWaveformMetadata` initializes `samplesStart` to `0` by default.

	References
	----------
	[1] `ArrayWaveforms`

	[2] `WaveformMetadata`
	"""
	sampleRateDesired = sampleRateDesired or setting.sampleRate
	dictionaryWaveformMetadata, axis = getWaveformMetadata(listPathFilenames, sampleRateDesired)

	shapeArray: tuple[int, ...] = tuple(entry.size for entry in sorted(axis.values()))

	arrayWaveforms: ArrayWaveforms = numpy.zeros(shapeArray, dtype=setting.dtypeWaveform)

	for index, metadata in dictionaryWaveformMetadata.items():
		arrayWaveforms[:, metadata['samplesStart']:metadata['samplesStop'], index] = readAudioFile(metadata['pathFilename'], sampleRateDesired)

	return arrayWaveforms

def _getSpectrogram(waveform: Waveform, metadata: WaveformMetadata, sampleRateDesired: float, **parametersSTFT: Any) -> Spectrogram:
	"""I use this to load a single audio file into a pre-allocated waveform buffer and compute its spectrogram.

	(AI generated docstring)

	I use this shared subroutine inside `loadSpectrograms` to avoid reallocating a waveform
	buffer for each file. `_getSpectrogram` copies audio data from `metadata['pathFilename']`
	into the caller-provided `waveform` buffer at the position described by `metadata`, then
	computes `stft` with `sampleRateDesired` and any additional `parametersSTFT`. The caller
	must pass a fresh copy of the buffer for each iteration.

	Parameters
	----------
	waveform : Waveform
		Pre-allocated buffer into which audio data is copied before the STFT. The caller
		must pass a separate copy for each file to prevent data from accumulating across
		iterations.
	metadata : WaveformMetadata
		Record describing `pathFilename`, `lengthWaveform`, `samplesStart`, and
		`samplesStop` for the audio file being loaded.
	sampleRateDesired : float
		Target sample rate passed to `readAudioFile`.
	**parametersSTFT : Any
		Keyword parameters forwarded to `stft`.

	Returns
	-------
	spectrogram : Spectrogram
		Complex spectrogram of `waveform` after copying the audio file into the buffer.

	"""
	waveform[:, metadata['samplesStart']:metadata['samplesStop']] = readAudioFile(metadata['pathFilename'], sampleRateDesired)
	return stft(waveform, sampleRate=sampleRateDesired, **parametersSTFT)

def loadSpectrograms(listPathFilenames: Sequence[FileDescriptorOrPath], sampleRateDesired: float | None = None, **parametersSTFT: Any) -> tuple[ArraySpectrograms, dict[int, WaveformMetadata]]:
	"""Load spectrograms from a list of audio files.

	You can use this function to batch-convert audio files to spectrograms in a single call.
	`loadSpectrograms` reads each file, pads all waveforms to the same length, computes the Short-Time
	Fourier Transform for each, and stacks the results into one `ArraySpectrograms` [1] array. The
	function also returns a `WaveformMetadata` [2] dictionary that describes each file's original
	length and padding.

	Parameters
	----------
	listPathFilenames : Sequence[FileDescriptorOrPath]
		List of paths to audio files.
	sampleRateDesired : float | None = None
		Target sample rate in Hz. Defaults to `44100` when `None`.
	**parametersSTFT : Any
		Keyword parameters forwarded to `stft`, such as `lengthWindowingFunction` and `lengthHop`.

	Returns
	-------
	tupleSpectrogramsMetadata : tuple[ArraySpectrograms, dict[int, WaveformMetadata]]
		A two-element `tuple`. The first element is `ArraySpectrograms` [1] shaped `(channels,
		frequencies, frames, countFiles)` as `complex64`. The second element is a `dict` mapping
		integer file indices to `WaveformMetadata` [2] records.
	"""
	# DEVELOPMENT `loadSpectrograms` is not an extension of `loadWaveforms` because each
	# `pathFilename` is transformed into a spectrogram: I don't create an intermediate
	# `arrayWaveforms`. Nevertheless, I want the functions to share as much logic as possible.
	sampleRateDesired = sampleRateDesired or setting.sampleRate

	dictionaryWaveformMetadata, axis = getWaveformMetadata(listPathFilenames, sampleRateDesired)

	waveformZeros: Waveform = numpy.zeros(shape=(axis['channel'].size, axis['time'].size), dtype=setting.dtypeWaveform)
	# TODO instead of creating an identifier, move the operation inside the call to initialize `arraySpectrograms`.
	spectrogramZeros: Spectrogram = stft(waveformZeros, sampleRate=sampleRateDesired, **parametersSTFT)

	arraySpectrograms: ArraySpectrograms = numpy.zeros(shape=(*spectrogramZeros.shape, len(dictionaryWaveformMetadata)), dtype=setting.dtypeSpectrogram)

	for index, metadata in tqdm(dictionaryWaveformMetadata.items()):
		arraySpectrograms[..., index] = _getSpectrogram(waveformZeros.copy(), metadata, sampleRateDesired, **parametersSTFT)

	return arraySpectrograms, dictionaryWaveformMetadata
