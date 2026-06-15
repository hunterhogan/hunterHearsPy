# pyright: reportAssignmentType=false
# pyright: reportUnnecessaryComparison=false
# pyright: reportUnusedVariable=false
# ty:ignore[invalid-assignment]
# ruff: noqa: DOC201
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hunterHearsPy import getAxis, readAudioFile, setting, stft, WaveformAxes, WaveformMetadata
from math import ceil as ceiling
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
		# TODO document that if `samplesPadding` is odd, the extra pad-sample is added to samplesStart.
		# TODO think about whether I want the extra pad-sample at the start or end.
		metadata['samplesStart'] = ceiling(samplesPadding * multiplicandSamplesStart)
		metadata['samplesStop'] = metadata['samplesStart'] + metadata['lengthWaveform']

	return dictionaryWaveformMetadata, axis

def loadWaveforms(listPathFilenames: Sequence[FileDescriptorOrPath], **keywordArguments: Any) -> ArrayWaveforms:
	"""Load a list of audio files into a single stacked NumPy array."""
	align: OptionsAlign = keywordArguments.get('align', setting.align)
	dtype: DTypeLike = keywordArguments.get('dtype', setting.dtypeWaveform)
	dtype_str: Options_dtype_str = keywordArguments.get('dtype_str', setting.dtype_str)
	sampleRateDesired: float = keywordArguments.get('sampleRate', setting.sampleRate)

	dictionaryWaveformMetadata, axis = getWaveformMetadata(listPathFilenames, sampleRateDesired, align)

	arrayWaveforms: ArrayWaveforms = numpy.zeros(tuple(entry.size for entry in sorted(axis.values())), dtype=dtype)

	def workhorse(index: int, metadata: WaveformMetadata) -> None:
		arrayWaveforms[:, metadata['samplesStart'] : metadata['samplesStop'], index] = readAudioFile(
			metadata['pathFilename'], sampleRateDesired, dtype_str
		).astype(dtype, copy=False)

	with ThreadPoolExecutor() as threadManager:
		tuple(tqdm(threadManager.map(workhorse, dictionaryWaveformMetadata, dictionaryWaveformMetadata.values()), total=len(dictionaryWaveformMetadata)))

	return arrayWaveforms

def _getSpectrogram(waveform: Waveform, metadata: WaveformMetadata, sampleRateDesired: float, dtype_str: Options_dtype_str, dtypeWaveform: DTypeLike, **parametersSTFT: Any) -> Spectrogram:
	waveform[:, metadata['samplesStart'] : metadata['samplesStop']] = readAudioFile(metadata['pathFilename'], sampleRateDesired, dtype_str).astype(dtypeWaveform, copy=False)
	# TODO Think about: this is one of the only places where padding with non-zero values could be desirable.
	return stft(waveform, **parametersSTFT)

def loadSpectrograms(listPathFilenames: Sequence[FileDescriptorOrPath], **keywordArguments: Any) -> tuple[ArraySpectrograms, dict[int, WaveformMetadata]]:
	"""Load spectrograms from a list of audio files."""
	# DEVELOPMENT `loadSpectrograms` is not an extension of `loadWaveforms` because each
	# `pathFilename` is transformed into a spectrogram: I don't create an intermediate
	# `arrayWaveforms`. Nevertheless, I want the functions to share as much logic as possible.
	if 'align' in keywordArguments:
		align: OptionsAlign = keywordArguments.pop('align')
	else:
		align = setting.align

	if 'dtype' in keywordArguments:
		dtype: DTypeLike = keywordArguments.pop('dtype')
	else:
		dtype = setting.dtypeSpectrogram

	if 'dtype_str' in keywordArguments:
		dtype_str: Options_dtype_str = keywordArguments.pop('dtype_str')
	else:
		dtype_str = setting.dtype_str

	if 'dtypeWaveform' in keywordArguments:
		dtypeWaveform: DTypeLike = keywordArguments.pop('dtypeWaveform')
	else:
		dtypeWaveform = setting.dtypeWaveform

	if 'sampleRateDesired' in keywordArguments:
		sampleRateDesired: float = keywordArguments.pop('sampleRateDesired')
	else:
		sampleRateDesired = setting.sampleRate

	dictionaryWaveformMetadata, axis = getWaveformMetadata(listPathFilenames, sampleRateDesired, align)

	waveformZeros: Waveform = numpy.zeros(shape=(axis['channel'].size, axis['time'].size), dtype=dtypeWaveform)

	# TODO instead of creating an identifier, move the operation inside the call to initialize `arraySpectrograms`.
	spectrogramZeros: Spectrogram = stft(waveformZeros, **keywordArguments)

	arraySpectrograms: ArraySpectrograms = numpy.zeros(shape=(*spectrogramZeros.shape, len(dictionaryWaveformMetadata)), dtype=dtype)

	for index, metadata in tqdm(dictionaryWaveformMetadata.items()):
		arraySpectrograms[..., index] = _getSpectrogram(waveformZeros.copy(), metadata, sampleRateDesired, dtype_str, dtypeWaveform, **keywordArguments)

	return arraySpectrograms, dictionaryWaveformMetadata
