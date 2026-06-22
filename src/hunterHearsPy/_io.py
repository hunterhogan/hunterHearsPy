from __future__ import annotations

from hunterHearsPy import amplitudeToSoundfile, getAxis, resampleWaveform, setting, stft
from hunterMakesPy.filesystemToolkit import makeDirectorySafely
from pathlib import Path, PurePath
from soundfile import AudioData
from typing import TYPE_CHECKING
from typing_extensions import Unpack
import numpy
import soundfile
import tempfile
import uuid

if TYPE_CHECKING:
	from hunterHearsPy import AxisMetadata, FileDescriptorOrPath, Parameters_stft, Spectrogram, Waveform
	from numpy import dtype, ndarray
	from soundfile import AudioData_2d, dtype_str as Options_dtype_str
	from typing import Any

def readAudioFile(pathFilename: FileDescriptorOrPath, sampleRateDesired: float | None = None, dtype_str: Options_dtype_str | None = None) -> Waveform:
	"""Read an audio file and return waveform data as a NumPy array.

	You can use this function to load any audio file that `soundfile` [1] supports. The returned
	`Waveform` [2] is always shaped `(channels, samples)`. When `sampleRateDesired` differs from the file's
	native sample rate, `readAudioFile` resamples using `resampleWaveform`.

	Parameters
	----------
	pathFilename : FileDescriptorOrPath
		Path to the audio file or a binary stream.
	sampleRateDesired : float | None = 44100
		Target sample rate of the returned `Waveform` [2] in Hz. Defaults to `setting.sampleRate`,
		which is probably 44100, when `None`.
	dtype_str : Literal["float64", "float32", "int32", "int16"] | None = 'float32'
		Data type for the returned `Waveform` [2]. Defaults to `setting.dtype_str`, which is probably 'float32', when `None`.

	Returns
	-------
	waveform : Waveform
		Audio data shaped `(channels, samples)` as `setting.dtypeWaveform`.
	"""
	sampleRateDesired = sampleRateDesired or setting.sampleRate
	with soundfile.SoundFile(pathFilename) as readSoundFile:
		sampleRateSource: int = readSoundFile.samplerate
		audioData: AudioData_2d = readSoundFile.read(dtype=dtype_str or setting.dtype_str, always_2d=True)  # ty:ignore[invalid-assignment] https://github.com/astral-sh/ty/issues/2799
	axis: dict[str, AxisMetadata] = getAxis()
	waveform: Waveform = audioData.transpose((axis['time'].number, axis['channel'].number))

	return resampleWaveform(waveform, sampleRateDesired, sampleRateSource, axis['time'].number)  # ty:ignore[no-matching-overload] https://github.com/astral-sh/ty/issues/2799

def writeWAV(pathFilename: FileDescriptorOrPath, waveform: Waveform, sampleRate: float | None = None, subtype: str | None = None) -> FileDescriptorOrPath:
	"""Write a waveform array to a WAV file.

	You can use this function to save a `Waveform` [1] or any compatible NumPy array to a 32-bit float
	WAV file. `writeWAV` creates any missing parent directories before writing using
	`makeDirectorySafely` from `hunterMakesPy` [2].

	Parameters
	----------
	pathFilename : FileDescriptorOrPath
		Destination path for the WAV file, or a binary stream.
	waveform : Waveform
		Audio data shaped `(channels, samples)` or `(samples,)`.
	sampleRate : float | None = None
		Sample rate of `waveform` in Hz. Defaults to `setting.sampleRate`, which is probably 44100, when `None`.

	Returns
	-------
	pathFilename : FileDescriptorOrPath
		The `FileDescriptorOrPath` passed in `pathFilename`, which simplifies using a functional
		paradigm.

	File Overwrite and Format
	-------------------------
	`writeWAV` overwrites any existing file at `pathFilename` without prompting. All files are written
	as 32-bit float WAV using `soundfile.write` [3].

	References
	----------
	[1] `Waveform`

	[2] hunterMakesPy — makeDirectorySafely
		https://context7.com/hunterhogan/huntermakespy
	[3] soundfile — audio library based on libsndfile
		https://python-soundfile.readthedocs.io/en/0.12.1/
	"""
	sampleRate = int(sampleRate or setting.sampleRate)
	subtype = subtype or setting.subtype
	makeDirectorySafely(pathFilename)

	audioData: AudioData = amplitudeToSoundfile(waveform)

	audioData = audioData.transpose()

	soundfile.write(file=pathFilename, data=audioData, samplerate=sampleRate, subtype=subtype, format='WAV')
	return pathFilename

def spectrogramToWAV(spectrogram: Spectrogram, pathFilename: FileDescriptorOrPath, lengthWaveform: int, **parametersSTFT: Unpack[Parameters_stft]) -> None:
	"""Write a complex spectrogram to a WAV file by computing the inverse STFT.

	You can use this function to reconstruct a waveform from a `Spectrogram` [1] and save
	it directly to a WAV file. `spectrogramToWAV` calls `stft` with `inverse=True` to
	obtain the reconstructed `Waveform` [2], then passes it to `writeWAV`.

	Parameters
	----------
	spectrogram : Spectrogram
		Complex spectrogram to convert back to a waveform.
	pathFilename : FileDescriptorOrPath
		Destination path for the WAV file, or a binary stream.
	lengthWaveform : int
		Number of samples in the output waveform. The inverse STFT cannot recover the
		original length from the spectrogram alone, so `lengthWaveform` is required.
	sampleRate : float | None = None
		Sample rate for the output WAV file in Hz. Defaults to `44100` when `None`.
	**parametersSTFT : Any
		Keyword parameters forwarded to `stft`, such as `lengthWindowingFunction` and
		`lengthHop`.

	File Overwrite and Format
	-------------------------
	See `writeWAV` for file overwrite behavior and output format details.

	References
	----------
	[1] `Spectrogram`

	[2] `Waveform`

	"""
	waveform: Waveform = stft(spectrogram, lengthWaveform=lengthWaveform, **parametersSTFT)
	sampleRate: float = parametersSTFT.get('sampleRate', setting.sampleRate)
	writeWAV(pathFilename, waveform, sampleRate)

def saveOnError(arrayTarget: ndarray[tuple[Any, ...], dtype[Any]], *, identifierTarget: str = 'arrayTarget') -> PurePath:
	pathFilename: Path = Path(tempfile.mkdtemp(prefix='hunterHearsPy'), f"{identifierTarget}_{uuid.uuid4().hex}.npy").resolve()
	numpy.save(pathFilename, arrayTarget)

	return PurePath(pathFilename)
