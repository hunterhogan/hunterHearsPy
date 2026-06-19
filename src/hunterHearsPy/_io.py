# pyright: reportArgumentType=false
# ruff: noqa: RUF069 ERA001
# ty:ignore[invalid-assignment]
from __future__ import annotations

from hunterHearsPy import ArraySpectrograms, ArrayWaveforms, getAxis, resampleWaveform, setting, Spectrogram, stft, Waveform
from hunterHearsPy.theSSOT import subtypeHARDCODED
from hunterMakesPy.filesystemToolkit import makeDirectorySafely
from pathlib import Path
from soundfile import dtype_str as Options_dtype_str
from typing import Any, TYPE_CHECKING
import numpy
import soundfile
import tempfile
import uuid

if TYPE_CHECKING:
	from hunterHearsPy import FileDescriptorOrPath, WaveformAxes

# TODO. The typing has too many moving parts.
# 1. This function: that should be the easiest--if I get the other parts right.
# 2. `SoundFile.read(dtype=setting.dtype_str` is dynamic, and the static checker is using the type of
#    the field, `dtype_str: soundfile_dtype_str`, `Literal["float64", "float32", "int32", "int16"]`
# 3. `Soundfile.read` has decent typing, but I have total control because I have a custom stub file in
#    `stubFileNotFound`.
# 4. `resampy` accepts integer or floating input and returns float32 or the same floating type.

# AND NOW, `writeWAV` is pulled into the problem.

# This is a general purpose function: it is not a subroutine of _arrays.
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
		waveform: Waveform = readSoundFile.read(dtype=dtype_str or setting.dtype_str, always_2d=True)
	axis: dict[str, WaveformAxes] = getAxis()
	waveform = waveform.transpose((axis['time'].number, axis['channel'].number))

	if float(sampleRateSource) != sampleRateDesired:
		# calling resampling will force int to float, so if the user wants int, sending it through resampling is NOT a no-op.
		# Therefore, this guard is necessary.
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
	makeDirectorySafely(pathFilename)
	axis: dict[str, WaveformAxes] = getAxis()
	# TODO what happens when waveform.shape is (samples,)? Do I care? waveform is currently typed as
	# ndarray[tuple[int, int], dtype[WaveformDtype]]. `waveform = waveform.T` is safe but not self-documenting.
	waveform = waveform.transpose((axis['time'].number, axis['channel'].number))

	# TODO Expand subtype in universal parameters and in the function parameters.
	subtype: str = subtypeHARDCODED
	# TODO: this is complicated.
	# ValueError: dtype must be one of ['float32', 'float64', 'int16', 'int32'] and not 'float16'
	# WaveformDtype: TypeAlias = floating[Any] | integer[Any]
	# Waveform: TypeAlias = ndarray[tuple[int, int], dtype[WaveformDtype]]
	soundfile.write(file=pathFilename, data=waveform, samplerate=sampleRate, subtype=subtype, format='WAV')
	return pathFilename

def spectrogramToWAV(spectrogram: Spectrogram, pathFilename: FileDescriptorOrPath, lengthWaveform: int, **parametersSTFT: Any) -> None:
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

def saveOnError(arrayTarget: Waveform | ArrayWaveforms | Spectrogram | ArraySpectrograms) -> str:
	pathFilename: Path = Path(tempfile.mkdtemp(prefix='hunterHearsPy'), f"arrayTarget_{uuid.uuid4().hex}.npy").resolve()
	numpy.save(pathFilename, arrayTarget)
	message: str = (
	"I did not receive `lengthWaveform`, so I could not perform the inverse STFT. "
	f"I saved `arrayTarget` to a file in this computer's temporary directory so you might recover the data. {arrayTarget.shape = }, {arrayTarget.dtype = }\n"
	f"{pathFilename = }"
	)

	return message
