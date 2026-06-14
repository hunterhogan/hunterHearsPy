# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# ruff: noqa: RUF069
# ty:ignore[invalid-assignment]
from __future__ import annotations

from hunterHearsPy import getAxis, resampleWaveform, setting
from hunterHearsPy.theSSOT import subtypeHARDCODED
from hunterMakesPy.filesystemToolkit import makeDirectorySafely
from typing import TYPE_CHECKING
import soundfile

if TYPE_CHECKING:
	from hunterHearsPy import FileDescriptorOrPath, Waveform
	from hunterHearsPy.theTypes import WaveformAxes

# TODO. The typing has too many moving parts.
# 1. This function: that should be the easiest--if I get the other parts right.
# 2. `SoundFile.read(dtype=setting.dtype_str` is dynamic, and the static checker is using the type of
#    the field, `dtype_str: soundfile_dtype_str`, `Literal["float64", "float32", "int32", "int16"]`
# 3. `Soundfile.read` has decent typing, but I have total control because I have a custom stub file in
#    `stubFileNotFound`.
# 4. the `resampy` types, which was ironically annotated by me, demands float input and returns float
#    output, but I now realize that I didn't test anything or look at the code.

# This is a general purpose function: it is not a subroutine of _arrays.
def readAudioFile(pathFilename: FileDescriptorOrPath, sampleRateDesired: float | None = None) -> Waveform:
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
		which is probably 44100 when `None`.

	Returns
	-------
	waveform : Waveform
		Audio data shaped `(channels, samples)` as `setting.dtypeWaveform`.
	"""
	sampleRateDesired = sampleRateDesired or setting.sampleRate
	with soundfile.SoundFile(pathFilename) as readSoundFile:
		sampleRateSource: int = readSoundFile.samplerate
		waveform: Waveform = readSoundFile.read(dtype=setting.dtype_str, always_2d=True)
	axis: dict[str, WaveformAxes] = getAxis()
	waveform = waveform.transpose((axis['time'].number, axis['channel'].number))

	if float(sampleRateSource) != sampleRateDesired:
		waveform = resampleWaveform(waveform, sampleRateDesired, sampleRateSource, axis['time'].number)

	return waveform

def writeWAV(pathFilename: FileDescriptorOrPath, waveform: Waveform, sampleRate: float | None = None) -> FileDescriptorOrPath:
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
		Sample rate of `waveform` in Hz. Defaults to `setting.sampleRate`, which is probably 44100
		when `None`.

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
	makeDirectorySafely(pathFilename)
	axis: dict[str, WaveformAxes] = getAxis()
	waveform = waveform.transpose((axis['time'].number, axis['channel'].number))

	# TODO Expand subtype in universal parameters and in the function parameters.
	subtype: str = subtypeHARDCODED
	soundfile.write(file=pathFilename, data=waveform, samplerate=sampleRate, subtype=subtype, format='WAV')
	return pathFilename
