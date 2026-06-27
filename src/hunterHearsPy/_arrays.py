# ruff: noqa: DOC201
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hunterHearsPy import (
	AxisMetadata, getAxis, readAudioFile, setting, SpectrogramsAndMetadata, stft, WaveformMetadata, WaveformsAndMetadata)
from hunterMakesPy.parseParameters import defineConcurrencyLimit
from tqdm.auto import tqdm
from typing import NamedTuple, TYPE_CHECKING
from typing_extensions import Unpack
import numpy
import sys

if TYPE_CHECKING:
	from collections.abc import Sequence
	from hunterHearsPy.theTypes import (
		ArraySpectrograms, ArrayWaveforms, FileDescriptorOrPath, OptionsAlign, Parameters_loadSpectrograms, Parameters_loadWaveforms, Waveform)
	from numpy.lib._arraypad_impl import _ModeKind
	from numpy.typing import DTypeLike
	from soundfile import dtype_str as Options_dtype_str

#======== Idiosyncratic classes tell type checkers about the shape of arrays ======================

class Axes2(NamedTuple):
	a0: int
	a1: int

class Axes3(NamedTuple):
	a0: int
	a1: int
	a2: int

class Axes4(NamedTuple):
	a0: int
	a1: int
	a2: int
	a3: int

def getWaveformMetadata(
	listPathFilenames: Sequence[FileDescriptorOrPath], sampleRate: float, align: OptionsAlign
) -> tuple[dict[int, WaveformMetadata], dict[str, AxisMetadata]]:
	"""Retrieve metadata for a collection of audio waveform files."""
	#============== Initialize ==========================================================

	axis: dict[str, AxisMetadata] = getAxis()
	channelMaximum: int = 0
	dictionaryWaveformMetadata: dict[int, WaveformMetadata] = {}
	lengthMaximum: int = 0

	#============== Populate ===========================================================

	if len(listPathFilenames) == 0:
		message: str = f'I received `{len(listPathFilenames) = }`, so `arrayWaveforms` will have zero-sized axes.'
		sys.stderr.write(message + '\n')

	for index, pathFilename in enumerate(tqdm(listPathFilenames, desc='Preparing combined array', leave=False)):
		# NOTE Use readAudioFile with the prescribed sampleRate to get the exact length.
		channels, lengthWaveform = readAudioFile(pathFilename, sampleRate).shape
		dictionaryWaveformMetadata[index] = WaveformMetadata(
			channels=channels, lengthWaveform=lengthWaveform, pathFilename=pathFilename, samplesStart=0, samplesStop=0
		)
		channelMaximum = max(channelMaximum, channels)
		lengthMaximum = max(lengthMaximum, lengthWaveform)

	axis['channel'] = AxisMetadata(number=axis['channel'].number, size=channelMaximum)
	axis['indexing'] = AxisMetadata(number=axis['indexing'].number, size=len(listPathFilenames))
	axis['time'] = AxisMetadata(number=axis['time'].number, size=lengthMaximum)

	#============== Calculate ===========================================================

	multiplicandSamplesStart: float = max((align == 'center') / 2, align == 'stop')

	for metadata in dictionaryWaveformMetadata.values():
		samplesPadding: int = axis['time'].size - metadata['lengthWaveform']
		# DOCUMENT that if `samplesPadding` is odd, the extra pad-sample is added to samplesStop.
		metadata['samplesStart'] = int(samplesPadding * multiplicandSamplesStart)
		metadata['samplesStop'] = metadata['samplesStart'] + metadata['lengthWaveform']

	return dictionaryWaveformMetadata, axis

def loadWaveforms(
	listPathFilenames: Sequence[FileDescriptorOrPath]
	, *
	, CPUlimit: bool | float | int | None = None
	, **keywordArguments: Unpack[Parameters_loadWaveforms]
) -> WaveformsAndMetadata:
	"""Load a list of audio files into a single stacked NumPy array."""
	#============== Initialize ==========================================================

	align: OptionsAlign = keywordArguments.get('align', setting.align)
	dtypeWaveform: DTypeLike = keywordArguments.get('dtypeWaveform', setting.dtypeWaveform)
	dtype_str: Options_dtype_str = keywordArguments.get('dtype_str', setting.dtype_str)
	sampleRate: float = keywordArguments.get('sampleRate', setting.sampleRate)

	max_workers: int = defineConcurrencyLimit(limit=CPUlimit)

	#============== Allocate memory ==========================================================

	dictionaryWaveformMetadata, axis = getWaveformMetadata(listPathFilenames, sampleRate, align)
	arrayWaveforms: ArrayWaveforms = numpy.zeros(Axes3(*(entry.size for entry in sorted(axis.values()))), dtypeWaveform)

	#============== Concurrent loading ==========================================================

	# TODO the axis order is hardcoded.
	def workhorse(index: int, metadata: WaveformMetadata) -> None:
		arrayWaveforms[:, metadata['samplesStart'] : metadata['samplesStop'], index] = readAudioFile(
			metadata['pathFilename'], sampleRate, dtype_str
		).astype(dtypeWaveform, copy=False)

	with ThreadPoolExecutor(max_workers=max_workers) as threadManager:
		tuple(tqdm(threadManager.map(workhorse, dictionaryWaveformMetadata, dictionaryWaveformMetadata.values())
				, desc='Loading waveforms.'
				, total=len(dictionaryWaveformMetadata)
		))

	return WaveformsAndMetadata(array=arrayWaveforms, metadata=dictionaryWaveformMetadata)

def loadSpectrograms(
	listPathFilenames: Sequence[FileDescriptorOrPath], *, CPUlimit: bool | float | int | None = None, **keywordArguments: Unpack[Parameters_loadSpectrograms]
) -> SpectrogramsAndMetadata:
	"""Load spectrograms from a list of audio files."""
	#============== Initialize ==========================================================

	align: OptionsAlign = keywordArguments.get('align', setting.align)
	align_pad_mode: _ModeKind = keywordArguments.get('align_pad_mode', setting.align_pad_mode)
	dtypeSpectrogram: DTypeLike = keywordArguments.get('dtypeSpectrogram', setting.dtypeSpectrogram)
	dtype_str: Options_dtype_str = keywordArguments.get('dtype_str', setting.dtype_str)
	dtypeWaveform: DTypeLike = keywordArguments.get('dtypeWaveform', setting.dtypeWaveform)
	sampleRate: float = keywordArguments.get('sampleRate', setting.sampleRate)

	max_workers: int = defineConcurrencyLimit(limit=CPUlimit)

	#============== Allocate memory ==========================================================

	dictionaryWaveformMetadata, axis = getWaveformMetadata(listPathFilenames, sampleRate, align)

	waveformShape = Axes2(*(entry.size for entry in sorted((axis['channel'], axis['time']))))
	axisSpectrogram = AxisMetadata(setting.axisSpectrogramIndexing, len(dictionaryWaveformMetadata))
	arraySpectrogramsShape: list[int] = list(stft(numpy.zeros(waveformShape, dtypeWaveform), **keywordArguments).shape)
	arraySpectrogramsShape.insert(axisSpectrogram.number, axisSpectrogram.size)

	arraySpectrograms: ArraySpectrograms = numpy.zeros(Axes4(*arraySpectrogramsShape), dtypeSpectrogram)

	#============== Concurrent loading ==========================================================

	def workhorse(index: int, metadata: WaveformMetadata) -> None:
		# TODO make tests that can check if this works.

		pad_width: dict[int, tuple[int, int]] = {
			axis['channel'].number: (0, 0)
			, axis['time'].number: (metadata['samplesStart'], axis['time'].size - metadata['samplesStop'])
		}

		# python3.10: TypeError: `pad_width` must be of integral type.
		# f.m.l.
		# TODO in 2026 October: drop Py3.10 and this crap. I just want semantic, dynamic access to the data! It's the year 2026, ffs!

		pad_widthIntegralType: tuple[tuple[int, int], ...] = tuple(dict(sorted(pad_width.items())).values())

		# TODO axisSpectrogram.number is hardcoded here. grr!
		arraySpectrograms[..., index] = stft(
			numpy.pad(
				readAudioFile(metadata['pathFilename'], sampleRate, dtype_str).astype(dtypeWaveform, copy=False)
				, pad_widthIntegralType
				, align_pad_mode
			)
			, **keywordArguments
		)

		if False:  # This fails the current tests.
			arraySpectrogramsShape[axisSpectrogram.number] = index
			indices = numpy.indices(tuple(arraySpectrogramsShape))
			numpy.put_along_axis(
				arraySpectrograms,
				indices=indices,
				values=stft(
					numpy.pad(
						readAudioFile(metadata['pathFilename'], sampleRate, dtype_str).astype(dtypeWaveform, copy=False),
						pad_widthIntegralType,
						mode=align_pad_mode,
					),
					**keywordArguments,
				),
				axis=axisSpectrogram.number,
			)

	with ThreadPoolExecutor(max_workers=max_workers) as threadManager:
		tuple(tqdm(threadManager.map(workhorse, dictionaryWaveformMetadata.keys(), dictionaryWaveformMetadata.values())
				, desc='Loading spectrograms.'
				, total=axisSpectrogram.size
		))

	return SpectrogramsAndMetadata(arraySpectrograms, dictionaryWaveformMetadata)

def BACKUPloadSpectrograms(
	listPathFilenames: Sequence[FileDescriptorOrPath], *, CPUlimit: bool | float | int | None = None, **keywordArguments: Unpack[Parameters_loadSpectrograms]
) -> tuple[ArraySpectrograms, dict[int, WaveformMetadata]]:
	"""Load spectrograms from a list of audio files."""
	#============== Initialize ==========================================================

	align: OptionsAlign = keywordArguments.get('align', setting.align)
	dtypeSpectrogram: DTypeLike = keywordArguments.get('dtypeSpectrogram', setting.dtypeSpectrogram)
	dtype_str: Options_dtype_str = keywordArguments.get('dtype_str', setting.dtype_str)
	dtypeWaveform: DTypeLike = keywordArguments.get('dtypeWaveform', setting.dtypeWaveform)
	sampleRate: float = keywordArguments.get('sampleRate', setting.sampleRate)

	max_workers: int = defineConcurrencyLimit(limit=CPUlimit)

	#============== Allocate memory ==========================================================

	dictionaryWaveformMetadata, axis = getWaveformMetadata(listPathFilenames, sampleRate, align)

	waveformShape = Axes2(*(entry.size for entry in sorted((axis['channel'], axis['time']))))
	waveformZeros: Waveform = numpy.zeros(waveformShape, dtypeWaveform)

	axisSpectrogram = AxisMetadata(setting.axisSpectrogramIndexing, len(dictionaryWaveformMetadata))

	arraySpectrogramsShape: list[int] = list(stft(waveformZeros, **keywordArguments).shape)
	arraySpectrogramsShape.insert(axisSpectrogram.number, axisSpectrogram.size)
	arraySpectrograms: ArraySpectrograms = numpy.zeros(Axes4(*arraySpectrogramsShape), dtypeSpectrogram)

	#============== Concurrent loading ==========================================================

	def workhorse(index: int, metadata: WaveformMetadata) -> None:
		waveform: Waveform = waveformZeros.copy()
		waveform[:, metadata['samplesStart'] : metadata['samplesStop']] = readAudioFile(
			metadata['pathFilename'], sampleRate, dtype_str
		).astype(dtypeWaveform, copy=False)
		arraySpectrograms[..., index] = stft(waveform, **keywordArguments)

	with ThreadPoolExecutor(max_workers=max_workers) as threadManager:
		tuple(
			tqdm(
				threadManager.map(workhorse, dictionaryWaveformMetadata.keys(), dictionaryWaveformMetadata.values()),
				desc='Loading spectrograms',
				total=axisSpectrogram.size,
			)
		)

	return arraySpectrograms, dictionaryWaveformMetadata
