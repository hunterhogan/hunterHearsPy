# ty:ignore[invalid-assignment]
# TODO If the typing stays like it is, I think ^^^ this is wrong.

from __future__ import annotations

from hunterHearsPy import FileDescriptorOrPath, resampleWaveform, setting
from hunterMakesPy.filesystemToolkit import makeDirectorySafely
from multiprocessing import set_start_method as multiprocessing_set_start_method
from typing import TYPE_CHECKING
import soundfile

if TYPE_CHECKING:
	from hunterHearsPy import Waveform

if __name__ == '__main__':
	multiprocessing_set_start_method('spawn')

# TODO. The typing has too many moving parts.
# 1. This function: that should be the easiest--if I get the other parts right.
# 2. `SoundFile.read(dtype=setting.dtype_str` is dynamic, and the static checker is using the type of
#    the field, `dtype_str: soundfile_dtype_str`, `Literal["float64", "float32", "int32", "int16"]`
# 3. `Soundfile.read` has decent typing, but I have total control because I have a custom stub file in
#    `stubFileNotFound`.
# 4. `resampy` only returns float dtype

# This is a general purpose function: it is not a subroutine of _arrays.
def readAudioFile(pathFilename: FileDescriptorOrPath, sampleRate: float | None = None) -> Waveform:
	"""Read an audio file and return waveform data as a NumPy array.

	You can use this function to load any audio file that `soundfile` [1] supports. The returned
	`Waveform` [2] is always shaped `(channels, samples)`. When `sampleRate` differs from the file's
	native sample rate, `readAudioFile` resamples using `resampleWaveform`.

	Parameters
	----------
	pathFilename : FileDescriptorOrPath
		Path to the audio file or a binary stream.
	sampleRate : float | None = 44100
		Target sample rate of the returned `Waveform` [2] in Hz. Defaults to `setting.sampleRate`,
		which is probably 44100 when `None`.

	Returns
	-------
	waveform : Waveform
		Audio data shaped `(channels, samples)` as `setting.dtypeWaveform`.
	"""
	sampleRate = sampleRate or setting.sampleRate
	with soundfile.SoundFile(pathFilename) as readSoundFile:
		sampleRateSource: int = readSoundFile.samplerate
		waveform: Waveform = readSoundFile.read(dtype=setting.dtype_str, always_2d=True)
	waveform = waveform.T  # Transpose to shape (channels, samples).

	return resampleWaveform(waveform, sampleRateDesired=sampleRate, sampleRateSource=sampleRateSource)

def writeWAV(pathFilename: FileDescriptorOrPath, waveform: Waveform, sampleRate: float | None = None) -> None:
	"""Write a waveform array to a WAV file.

	You can use this function to save a `Waveform` [1] or any compatible NumPy array to a
	32-bit float WAV file. `writeWAV` creates any missing parent directories before writing
	using `makeDirectorySafely` from `hunterMakesPy` [2].

	Parameters
	----------
	pathFilename : FileDescriptorOrPath
		Destination path for the WAV file, or a binary stream.
	waveform : Waveform
		Audio data shaped `(channels, samples)` or `(samples,)`.
	sampleRate : float | None = None
		Sample rate of `waveform` in Hz. Defaults to `setting.sampleRate`, which is probably 44100 when `None`.

	File Overwrite and Format
	-------------------------
	`writeWAV` overwrites any existing file at `pathFilename` without prompting. All files
	are written as 32-bit float WAV using `soundfile.write` [3].

	References
	----------
	[1] `Waveform`

	[2] hunterMakesPy — makeDirectorySafely
		https://context7.com/hunterhogan/huntermakespy
	[3] soundfile — audio library based on libsndfile
		https://python-soundfile.readthedocs.io/en/0.12.1/
	"""
	sampleRate = sampleRate or setting.sampleRate
	makeDirectorySafely(pathFilename)
	# TODO Expand subtype in universal parameters and in the function parameters.
	soundfile.write(file=pathFilename, data=waveform.T, samplerate=int(sampleRate), subtype='FLOAT', format='WAV')
