# ruff: noqa: DOC201
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hunterHearsPy import (
	ArraySpectrogramsShape, ArrayWaveformsShape, AxisMetadata, getAxis, readAudioFile, setting, stft, WaveformMetadata, WaveformShape)
from hunterMakesPy.parseParameters import defineConcurrencyLimit
from tqdm.auto import tqdm
from typing import TYPE_CHECKING
from typing_extensions import Unpack
import numpy
import sys

if TYPE_CHECKING:
	from collections.abc import Sequence
	from hunterHearsPy import (
		ArraySpectrograms, ArrayWaveforms, FileDescriptorOrPath, OptionsAlign, Parameters_loadSpectrograms, Parameters_loadWaveforms, Waveform)
	from numpy.lib._arraypad_impl import _ModeKind
	from numpy.typing import DTypeLike
	from soundfile import dtype_str as Options_dtype_str
	from typing import Any

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
		# TODO document that if `samplesPadding` is odd, the extra pad-sample is added to samplesStop.
		metadata['samplesStart'] = int(samplesPadding * multiplicandSamplesStart)
		metadata['samplesStop'] = metadata['samplesStart'] + metadata['lengthWaveform']

	return dictionaryWaveformMetadata, axis

def loadWaveforms(listPathFilenames: Sequence[FileDescriptorOrPath], *, CPUlimit: bool | float | int | None = None, **keywordArguments: Unpack[Parameters_loadWaveforms]) -> ArrayWaveforms:
	"""Load a list of audio files into a single stacked NumPy array."""
	#============== Initialize ==========================================================
	align: OptionsAlign = keywordArguments.get('align', setting.align)
	dtype: DTypeLike = keywordArguments.get('dtype', setting.dtypeWaveform)
	dtype_str: Options_dtype_str = keywordArguments.get('dtype_str', setting.dtype_str)
	sampleRateDesired: float = keywordArguments.get('sampleRateDesired', setting.sampleRate)

	max_workers: int = defineConcurrencyLimit(limit=CPUlimit)

	#============== Allocate memory ==========================================================
	dictionaryWaveformMetadata, axis = getWaveformMetadata(listPathFilenames, sampleRateDesired, align)

	arrayWaveforms: ArrayWaveforms = numpy.zeros(ArrayWaveformsShape(*(entry.size for entry in sorted(axis.values()))), dtype)

	#============== Concurrent loading ==========================================================
	# TODO the axis order is hardcoded.
	def workhorse(index: int, metadata: WaveformMetadata) -> None:
		arrayWaveforms[:, metadata['samplesStart'] : metadata['samplesStop'], index] = readAudioFile(
			metadata['pathFilename'], sampleRateDesired, dtype_str
		).astype(dtype, copy=False)

	with ThreadPoolExecutor(max_workers=max_workers) as threadManager:
		tuple(tqdm(threadManager.map(workhorse, dictionaryWaveformMetadata, dictionaryWaveformMetadata.values()), total=len(dictionaryWaveformMetadata)))

	"""# DEVELOPMENT
	test_loadWaveforms_align: this is an exception to the rule of 1 test function for non-error outcomes per function. The not-commented-out code in align.py is my proof of concept for this test. But I DO NOT want you to mimic the code I wrote. I want you to understand what we want to test, then I want you think about a generalized test function. I designed this ArrayWaveforms such that if all of the waveforms are align=start (which means "pad the end of the shorter waveforms"), then for the duration of the shortest waveform, that data is identical in all of the waveforms. In this case, the shortest is 20 seconds at 44100 Hz, so the first 44100*20 samples of the left channel in all 5 waveforms match each other and the first 44100*20 samples of the right channel in all 5 waveforms match each other. Said, differently, if you truncate to the shortest waveform, then all of the waveforms should be identical.

	If align=stop, then if you truncate to the shortest waveform by removing the beginning of the longer waveforms, then all waveforms are identical.

	If align=center, truncate by removing equal amounts from the beginning and end, and the waveforms are identical.

	That is the test I want you to make. The first three parameters are the same as test_loadWaveforms because they are necessary to get the array. Then use the align parameter in the test function to truncate/trim the array to the shortest waveform. Then do a round-robin assert_array_equal for each waveform in the ArrayWaveforms.

	How long is the shortest waveform? Well, fuck.

	# TODO Do I need to return `dictionaryWaveformMetadata`? In addition to the padding data, it is a convenient way to get the original pathFilename.
	I wrote a bunch of notes about this. I guess they are in old commits.
	Instead of returning a simple tuple, I could return a named tuple, which is tuple+, of course.
	"""

	return arrayWaveforms

def loadSpectrograms(listPathFilenames: Sequence[FileDescriptorOrPath], *, CPUlimit: bool | float | int | None = None, **keywordArguments: Any) -> tuple[ArraySpectrograms, dict[int, WaveformMetadata]]:
	"""Load spectrograms from a list of audio files."""
	#============== Initialize ==========================================================

	align: OptionsAlign = keywordArguments.get('align', setting.align)
	align_pad_mode: _ModeKind = keywordArguments.get('align_pad_mode', setting.align_pad_mode)
	dtype: DTypeLike = keywordArguments.get('dtype', setting.dtypeSpectrogram)
	dtype_str: Options_dtype_str = keywordArguments.get('dtype_str', setting.dtype_str)
	dtypeWaveform: DTypeLike = keywordArguments.get('dtypeWaveform', setting.dtypeWaveform)
	sampleRateDesired: float = keywordArguments.get('sampleRateDesired', setting.sampleRate)

	max_workers: int = defineConcurrencyLimit(limit=CPUlimit)

	#============== Allocate memory ==========================================================

	dictionaryWaveformMetadata, axis = getWaveformMetadata(listPathFilenames, sampleRateDesired, align)

	waveformShape = WaveformShape(*(entry.size for entry in sorted((axis['channel'], axis['time']))))
	axisSpectrogram = AxisMetadata(setting.axisSpectrogramIndexing, len(dictionaryWaveformMetadata))
	arraySpectrogramsShape: list[int] = list(stft(numpy.zeros(waveformShape, dtypeWaveform), **keywordArguments).shape)
	arraySpectrogramsShape.insert(axisSpectrogram.number, axisSpectrogram.size)

	arraySpectrograms: ArraySpectrograms = numpy.zeros(ArraySpectrogramsShape(*arraySpectrogramsShape), dtype)

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
				readAudioFile(metadata['pathFilename'], sampleRateDesired, dtype_str).astype(dtypeWaveform, copy=False)
				, pad_widthIntegralType, mode=align_pad_mode)
			, **keywordArguments
		)

		if False:  # This fails the current tests.
			arraySpectrogramsShape[axisSpectrogram.number] = index
			indices = numpy.indices(tuple(arraySpectrogramsShape))
			numpy.put_along_axis(arraySpectrograms, indices=indices, values=stft(
				numpy.pad(
					readAudioFile(metadata['pathFilename'], sampleRateDesired, dtype_str).astype(dtypeWaveform, copy=False)
					, pad_widthIntegralType, mode=align_pad_mode)
				, **keywordArguments
			), axis=axisSpectrogram.number
		)

	with ThreadPoolExecutor(max_workers=max_workers) as threadManager:
		tuple(tqdm(threadManager.map(workhorse, dictionaryWaveformMetadata.keys(), dictionaryWaveformMetadata.values()), desc='Loading spectrograms', total=axisSpectrogram.size))

	return arraySpectrograms, dictionaryWaveformMetadata

def BACKUPloadSpectrograms(listPathFilenames: Sequence[FileDescriptorOrPath], *, CPUlimit: bool | float | int | None = None, **keywordArguments: Any) -> tuple[ArraySpectrograms, dict[int, WaveformMetadata]]:
	"""Load spectrograms from a list of audio files."""
	#============== Initialize ==========================================================

	align: OptionsAlign = keywordArguments.get('align', setting.align)
	dtype: DTypeLike = keywordArguments.get('dtype', setting.dtypeSpectrogram)
	dtype_str: Options_dtype_str = keywordArguments.get('dtype_str', setting.dtype_str)
	dtypeWaveform: DTypeLike = keywordArguments.get('dtypeWaveform', setting.dtypeWaveform)
	sampleRateDesired: float = keywordArguments.get('sampleRateDesired', setting.sampleRate)

	max_workers: int = defineConcurrencyLimit(limit=CPUlimit)

	#============== Allocate memory ==========================================================

	dictionaryWaveformMetadata, axis = getWaveformMetadata(listPathFilenames, sampleRateDesired, align)

	waveformShape = WaveformShape(*(entry.size for entry in sorted((axis['channel'], axis['time']))))
	waveformZeros: Waveform = numpy.zeros(waveformShape, dtypeWaveform)

	axisSpectrogram = AxisMetadata(setting.axisSpectrogramIndexing, len(dictionaryWaveformMetadata))

	arraySpectrogramsShape: list[int] = list(stft(waveformZeros, **keywordArguments).shape)
	arraySpectrogramsShape.insert(axisSpectrogram.number, axisSpectrogram.size)
	arraySpectrograms: ArraySpectrograms = numpy.zeros(ArraySpectrogramsShape(*arraySpectrogramsShape), dtype)

	#============== Concurrent loading ==========================================================

	def workhorse(index: int, metadata: WaveformMetadata) -> None:
		waveform: Waveform = waveformZeros.copy()
		waveform[:, metadata['samplesStart'] : metadata['samplesStop']] = readAudioFile(metadata['pathFilename'], sampleRateDesired, dtype_str).astype(dtypeWaveform, copy=False)
		arraySpectrograms[..., index] = stft(waveform, **keywordArguments)

	with ThreadPoolExecutor(max_workers=max_workers) as threadManager:
		tuple(tqdm(threadManager.map(workhorse, dictionaryWaveformMetadata.keys(), dictionaryWaveformMetadata.values()), desc='Loading spectrograms', total=axisSpectrogram.size))

	return arraySpectrograms, dictionaryWaveformMetadata
