# pyright: reportUnnecessaryComparison=false
# pyright: reportUnusedVariable=false
# ty:ignore[invalid-assignment]
from __future__ import annotations

from hunterHearsPy import parametersUniversal, resampleWaveform, universalDtypeWaveform
from hunterMakesPy.filesystemToolkit import makeDirectorySafely
from multiprocessing import set_start_method as multiprocessing_set_start_method
from typing import cast, TYPE_CHECKING
import numpy
import soundfile

if TYPE_CHECKING:
	from hunterHearsPy import Waveform
	from os import PathLike
	from typing import Any, BinaryIO

if __name__ == '__main__':
	multiprocessing_set_start_method('spawn')

def readAudioFile(pathFilename: str | PathLike[Any] | BinaryIO, sampleRate: float | None = None) -> Waveform:
	"""Read an audio file and return stereo waveform data as a NumPy array.

	You can use this function to load any audio file that `soundfile` [1] supports. The returned
	`Waveform` [2] is always shaped `(channels, samples)` where `channels` is `2`. When the source
	file is mono, `readAudioFile` duplicates the single channel to produce a stereo array. When
	`sampleRate` differs from the file's native sample rate, `readAudioFile` resamples using
	`resampleWaveform`.

	Parameters
	----------
	pathFilename : str | PathLike[Any] | BinaryIO
		Path to the audio file or a binary stream compatible with `soundfile` [1].
	sampleRate : float | None = 44100
		Target sample rate of the returned `Waveform` [2] in Hz. Defaults to `44100` when `None`.

	Returns
	-------
	waveform : Waveform
		Stereo audio data shaped `(2, samples)` as `float32`.

	Raises
	------
	FileNotFoundError
		When `pathFilename` does not exist on the filesystem.
	RuntimeError
		When `pathFilename` is an unsupported or corrupted audio format.

	References
	----------
	[1] soundfile — audio library based on libsndfile
		https://python-soundfile.readthedocs.io/en/0.12.1/

	[2] `Waveform`

	"""
	if sampleRate is None:
		sampleRate = parametersUniversal['sampleRate']
	try:
		with soundfile.SoundFile(str(pathFilename)) as readSoundFile:
			sampleRateSource: int = readSoundFile.samplerate
			waveform: Waveform = readSoundFile.read(dtype='float32', always_2d=True).astype(universalDtypeWaveform)
	except soundfile.LibsndfileError as ERRORmessage:
		if 'System error' in str(ERRORmessage):
			message = f"File not found: {pathFilename}"
			raise FileNotFoundError(message) from ERRORmessage
		raise RuntimeError from ERRORmessage

	axisTime = 0
	axisChannels = 1
	waveform = cast('Waveform', resampleWaveform(waveform, sampleRateDesired=sampleRate, sampleRateSource=sampleRateSource, axisTime=axisTime))
	# TODO In my audio ecosystem, must I force a minimum of 2 channels, or can I merely force an axis for time, even if the axis is length=1?
	if waveform.shape[axisChannels] == 1:
		waveform = cast('Waveform', numpy.repeat(waveform, 2, axis=axisChannels))
	return cast('Waveform', numpy.transpose(waveform, axes=(axisChannels, axisTime)))

def writeWAV(pathFilename: str | PathLike[Any] | BinaryIO, waveform: Waveform, sampleRate: float | None = None) -> None:
	"""Write a waveform array to a WAV file.

	You can use this function to save a `Waveform` [1] or any compatible NumPy array to a
	32-bit float WAV file. `writeWAV` creates any missing parent directories before writing
	using `makeDirsSafely` from `hunterMakesPy` [2].

	Parameters
	----------
	pathFilename : str | PathLike[Any] | BinaryIO
		Destination path for the WAV file, or a binary stream.
	waveform : Waveform
		Audio data shaped `(channels, samples)` or `(samples,)`.
	sampleRate : float | None = None
		Sample rate of `waveform` in Hz. Defaults to `44100` when `None`.

	File Overwrite and Format
	-------------------------
	`writeWAV` overwrites any existing file at `pathFilename` without prompting. All files
	are written as 32-bit float WAV using `soundfile.write` [3].

	References
	----------
	[1] `Waveform`

	[2] hunterMakesPy — makeDirsSafely
		https://context7.com/hunterhogan/huntermakespy

	[3] soundfile — audio library based on libsndfile
		https://python-soundfile.readthedocs.io/en/0.12.1/

	"""
	if sampleRate is None:
		sampleRate = parametersUniversal['sampleRate']
	makeDirectorySafely(pathFilename)
	soundfile.write(file=pathFilename, data=waveform.T, samplerate=int(sampleRate), subtype='FLOAT', format='WAV')
