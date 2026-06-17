# ruff: noqa: DOC201
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hunterHearsPy import ArrayWaveformsShape, getAxis, readAudioFile, setting, stft, WaveformAxes, WaveformMetadata
from hunterMakesPy.parseParameters import defineConcurrencyLimit
from tqdm.auto import tqdm
from typing import TYPE_CHECKING
import numpy
import sys

if TYPE_CHECKING:
	from collections.abc import Sequence
	from hunterHearsPy import ArraySpectrograms, ArrayWaveforms, FileDescriptorOrPath, OptionsAlign, Spectrogram, Waveform
	from numpy.typing import DTypeLike
	from soundfile import dtype_str as Options_dtype_str
	from typing import Any

# TODO concurrency `loadSpectrograms`
# TODO more sophisticated tests

def getWaveformMetadata(
	listPathFilenames: Sequence[FileDescriptorOrPath], sampleRate: float, align: OptionsAlign
) -> tuple[dict[int, WaveformMetadata], dict[str, WaveformAxes]]:
	"""Retrieve metadata for a collection of audio waveform files."""
	# ======== Initialize ==========================================================

	axis: dict[str, WaveformAxes] = getAxis()
	channelMaximum: int = 0
	dictionaryWaveformMetadata: dict[int, WaveformMetadata] = {}
	lengthMaximum: int = 0

	# ======== Populate ===========================================================

	if len(listPathFilenames) == 0:
		message: str = f'I received `{len(listPathFilenames) = }`, so `arrayWaveforms` will have zero-sized axes.'
		sys.stderr.write(message + '\n')

	for index, pathFilename in enumerate(tqdm(listPathFilenames, desc='Preparing combined array', leave=False)):
		channels, lengthWaveform = readAudioFile(pathFilename, sampleRate).shape
		dictionaryWaveformMetadata[index] = WaveformMetadata(
			channels=channels, lengthWaveform=lengthWaveform, pathFilename=pathFilename, samplesStart=0, samplesStop=0
		)
		channelMaximum = max(channelMaximum, channels)
		lengthMaximum = max(lengthMaximum, lengthWaveform)

	axis['channel'] = WaveformAxes(number=axis['channel'].number, size=channelMaximum)
	axis['indexing'] = WaveformAxes(number=axis['indexing'].number, size=len(listPathFilenames))
	axis['time'] = WaveformAxes(number=axis['time'].number, size=lengthMaximum)

	# ======== Calculate ===========================================================

	multiplicandSamplesStart: float = max((align == 'center') / 2, align == 'start')

	for metadata in dictionaryWaveformMetadata.values():
		samplesPadding: int = axis['time'].size - metadata['lengthWaveform']
		# TODO document that if `samplesPadding` is odd, the extra pad-sample is added to samplesStop.
		metadata['samplesStart'] = int(samplesPadding * multiplicandSamplesStart)
		metadata['samplesStop'] = metadata['samplesStart'] + metadata['lengthWaveform']

	return dictionaryWaveformMetadata, axis

def loadWaveforms(listPathFilenames: Sequence[FileDescriptorOrPath], *, CPUlimit: bool | float | int | None = None, **keywordArguments: Any) -> ArrayWaveforms:
	"""Load a list of audio files into a single stacked NumPy array."""
	align: OptionsAlign = keywordArguments.get('align', setting.align)
	dtype: DTypeLike = keywordArguments.get('dtype', setting.dtypeWaveform)
	dtype_str: Options_dtype_str = keywordArguments.get('dtype_str', setting.dtype_str)
	sampleRateDesired: float = keywordArguments.get('sampleRate', setting.sampleRate)

	max_workers: int = defineConcurrencyLimit(limit=CPUlimit)

	dictionaryWaveformMetadata, axis = getWaveformMetadata(listPathFilenames, sampleRateDesired, align)

	arrayWaveforms: ArrayWaveforms = numpy.zeros(ArrayWaveformsShape(*(entry.size for entry in sorted(axis.values()))), dtype)
	# TODO frustrating! ^^^ in the line above, the axis order is entirely based on the SSOT, `axis`,
	# but IMMEDIATELY below, the axis order is hardcoded!

	def workhorse(index: int, metadata: WaveformMetadata) -> None:
		arrayWaveforms[:, metadata['samplesStart'] : metadata['samplesStop'], index] = readAudioFile(
			metadata['pathFilename'], sampleRateDesired, dtype_str
		).astype(dtype, copy=False)

	with ThreadPoolExecutor(max_workers=max_workers) as threadManager:
		tuple(tqdm(threadManager.map(workhorse, dictionaryWaveformMetadata, dictionaryWaveformMetadata.values()), total=len(dictionaryWaveformMetadata)))

	return arrayWaveforms

def loadSpectrograms(listPathFilenames: Sequence[FileDescriptorOrPath], *, CPUlimit: bool | float | int | None = None, **keywordArguments: Any) -> tuple[ArraySpectrograms, dict[int, WaveformMetadata]]:
	"""Load spectrograms from a list of audio files."""
	# DEVELOPMENT `loadSpectrograms` is not an extension of `loadWaveforms` because each
	# `pathFilename` is transformed into a spectrogram: I don't create an intermediate
	# `arrayWaveforms`. Nevertheless, I want the functions to share as much logic as possible.

	align: OptionsAlign = keywordArguments.get('align', setting.align)
	dtype: DTypeLike = keywordArguments.get('dtype', setting.dtypeSpectrogram)
	dtype_str: Options_dtype_str = keywordArguments.get('dtype_str', setting.dtype_str)
	dtypeWaveform: DTypeLike = keywordArguments.get('dtypeWaveform', setting.dtypeWaveform)
	sampleRateDesired: float = keywordArguments.get('sampleRateDesired', setting.sampleRate)

	max_workers: int = defineConcurrencyLimit(limit=CPUlimit)

	dictionaryWaveformMetadata, axis = getWaveformMetadata(listPathFilenames, sampleRateDesired, align)

	waveformZeros: Waveform = numpy.zeros(shape=(axis['channel'].size, axis['time'].size), dtype=dtypeWaveform)

	arraySpectrograms: ArraySpectrograms = numpy.zeros(shape=(*stft(waveformZeros, **keywordArguments).shape, len(dictionaryWaveformMetadata)), dtype=dtype)

	def workhorse(waveform: Waveform, metadata: WaveformMetadata, **parametersSTFT: Any) -> Spectrogram:
		waveform[:, metadata['samplesStart'] : metadata['samplesStop']] = readAudioFile(metadata['pathFilename'], sampleRateDesired, dtype_str).astype(dtypeWaveform, copy=False)
		# TODO Think about numpy.pad.mode waveform = numpy.pad(waveform, ((0, 0), (metadata['samplesStart'], waveform.shape[1] - metadata['samplesStop'])), mode=mode)
		return stft(waveform, **parametersSTFT)

	for index, metadata in tqdm(dictionaryWaveformMetadata.items()):
		arraySpectrograms[..., index] = workhorse(waveformZeros.copy(), metadata, **keywordArguments)

	return arraySpectrograms, dictionaryWaveformMetadata
