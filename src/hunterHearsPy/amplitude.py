# ruff: noqa: D103
"""Normalize audio waveform amplitudes.

(AI generated docstring)

You can use this module to scale audio waveforms to a target peak amplitude. Each normalization
function returns both the scaled waveform and a reversion callable that restores the original
amplitude scale when applied to any waveform derived from the normalized result.

Contents
--------
Functions
	normalizeArrayWaveforms
		Normalize multiple waveforms in an array to a specified peak amplitude.
	normalizeWaveform
		Normalize a waveform to a specified peak amplitude.

"""

from __future__ import annotations

from hunterMakesPy import zeroIndexed
from numpy import divide, finfo as numpy_finfo, float32, floating, iinfo as numpy_iinfo, integer, max as numpy_max, multiply
from soundfile import dtype_str
from typing import overload, TYPE_CHECKING
import numpy
import sys

if TYPE_CHECKING:
	from hunterHearsPy.theTypes import ArrayWaveforms, NormalizationReverter, Waveform, 形Shape
	from numpy import dtype, ndarray
	from pathlib import PurePath
	from soundfile import AudioData
	from typing import Any

def amplitudeIntegerToFloating(arrayTarget: ndarray[形Shape, dtype[integer[Any]]]) -> ndarray[形Shape, dtype[floating[Any]]]:
	# TODO Wait a minute. `iinfo` is "Machine limits for integer types."? Scaling should not be tied to the machine.
	integerInformation: numpy_iinfo[integer] = numpy_iinfo(arrayTarget.dtype.str)
	dtypeFloating: dtype[floating[Any]] = numpy.promote_types(arrayTarget.dtype, float32)
	arrayFloating: ndarray[形Shape, dtype[floating[Any]]] = numpy.astype(arrayTarget, dtypeFloating, copy=False)
	if integerInformation.min < 0:
		arrayFloating /= -integerInformation.min
	else:
		amplitudeMiddle: float = (integerInformation.max + zeroIndexed) / 2
		arrayFloating -= amplitudeMiddle
		arrayFloating /= amplitudeMiddle
	return arrayFloating

def amplitudeToSoundfile(arrayTarget: ndarray[形Shape, dtype[Any]]) -> AudioData:
	"""Handle funky dtypes.

	Like `resampleWaveform`, `writeWAV` is not restricted to "ecosystem" functions. It has far more
	limited output options than soundfile, for example, but I don't want to unnecessarily restrict
	input dtypes. I mean, if the user has a 2-axis `ndarray` of PCM data, then the only other
	criterion is having one of the four dtypes in `dtype_str`. There are only about 10 potential
	dtypes, including the 4 supported dtypes, that a user could have. Under a least-cost producer or
	comparative advantage rationale, I ought to create the logic that funnels the 10 potential types
	into the four support types, rather than forcing the user to do it. Therefore, this function
	exists. But it is a pain in the ass, people usually free-ride the least-cost producers, and the
	world certainly has not reciprocated my pro-social behavior.

	Returns
	-------
	arraySoundfile : AudioData
		The input array converted to a dtype compatible with `soundfile` [1] if necessary, otherwise
		returned unchanged.
	"""
	# Four dtype buckets
	# 1. `dtype_str`.
	# 2. integer, signed and unsigned.
	# 3. floating, single and double precision.
	# 4. other types I didn't anticipate.

	# Four options
	# 1. do nothing.
	# 2. shift unsigned to signed, and scale integer range.
	# 3. change the number of significant digits.
	# 4. attempt a forceful conversion to the most forgiving dtype in `dtype_str.__args__`.
	# ^^^ WAIT! If I don't know why this happening, I don't know if I need to scale or how to scale.
	# This isn't a normal `.astype` situation: the values are on a scale and changing the dtype can
	# change the scale.

	# Some of this stuff should be documented for AI agents and/or contributors. (ha!)

	# I do not use warning if I really want the user to see the message because
	# PyTorch spams so many warnings that many packages and people silence _all_ warnings.

	dtypeSoundfile: tuple[dtype[Any], ...] = tuple(map(numpy.dtype, dtype_str.__args__))
	dtypeMaximum = max(dtypeSoundfile)

	if arrayTarget.dtype.name in dtype_str.__args__:
		arraySoundfile: AudioData = arrayTarget

	elif numpy.issubdtype(arrayTarget.dtype, numpy.floating):
		dtypeSoundfileFloating: frozenset[dtype[floating[Any]]] = frozenset(filter(lambda _dtype: numpy.issubdtype(_dtype, floating), dtypeSoundfile))
		dtypeNewFloating: dtype[floating[Any]] = min(min(arrayTarget.dtype, *dtypeSoundfileFloating), max(dtypeSoundfileFloating))  # noqa: PLW3301
		arraySoundfile = numpy.astype(arrayTarget, dtypeNewFloating, copy=False)

	elif numpy.issubdtype(arrayTarget.dtype, numpy.integer):
		dtypesInteger: frozenset[dtype[integer[Any]]] = frozenset(filter(lambda _dtype: numpy.issubdtype(_dtype, integer), dtypeSoundfile))
		dtypeNewInteger: dtype[integer[Any]] = min(min(arrayTarget.dtype, *dtypesInteger), max(dtypesInteger))  # noqa: PLW3301

		integerInformation: numpy_iinfo[integer] = numpy_iinfo(arrayTarget.dtype)

		if integerInformation.min < 0:
			arrayTarget //= -integerInformation.min
		else:
			amplitudeMiddle: int = (integerInformation.max + zeroIndexed) // 2
			arrayTarget -= amplitudeMiddle
			arrayTarget //= amplitudeMiddle

		arraySoundfile = numpy.astype(arrayTarget, dtypeNewInteger, copy=False)  # pyright: ignore[reportAssignmentType]
	else:
		from hunterHearsPy._io import saveOnError  # noqa: PLC0415

		pathFilename: PurePath = saveOnError(arrayTarget)
		message: str = (
			f'Converting `arrayTarget` to a dtype in {dtype_str.__args__} may have failed or corrupted the data. '
			"I saved `arrayTarget` to a file in this computer's temporary directory if you need to recover the data. "
			f'{arrayTarget.shape = }, {arrayTarget.dtype = }\n'
			f'{pathFilename = }'
		)
		sys.stderr.write(message + '\n')

		# NOTE, I considered using contextlib.suppress, but then `arraySoundFile` would be
		# unassigned/unbound, so an Exception would occur at the return statement.
		arraySoundfile = numpy.astype(arrayTarget, dtypeMaximum, copy=False)

	# I could hardcode the potential types 'float32', 'float64', 'int16', 'int32' to make the type annotations align.
	# I don't know how, or if it is possible, to future-proof the type annotations against changes to `dtype_str.__args__`.
	return arraySoundfile  # ty:ignore[invalid-return-type]

def normalizeWaveform(waveform: Waveform, amplitudeNorm: float = 1.0) -> tuple[Waveform, NormalizationReverter]:
	"""Normalize a waveform to a specified peak amplitude.

	(AI generated docstring)

	You can use this function to scale a `Waveform` [1] so that its absolute peak value equals
	`amplitudeNorm`. This function also returns `revertNormalization`, a `NormalizationReverter` [2]
	callable that reverses the scaling when applied to any waveform derived from `waveformNormalized`.

	Parameters
	----------
	waveform : Waveform
		The input audio waveform to normalize.
	amplitudeNorm : float = 1.0
		Target peak amplitude. The absolute maximum value of `waveformNormalized` equals `amplitudeNorm`.

	Returns
	-------
	waveformNormalized : Waveform
		The scaled waveform with absolute peak value equal to `amplitudeNorm`.
	revertNormalization : NormalizationReverter
		A callable that reverses the normalization scaling. Apply `revertNormalization` to any
		waveform derived from `waveformNormalized` to restore the original amplitude scale.

	Warns
	-----
	UserWarning
		If `amplitudeNorm` is 0, `normalizeWaveform` replaces it with the smallest positive finite
		value representable in the dtype of `waveform` using `numpy.finfo` [3] and continues.
	UserWarning
		If `waveform` contains only zeros, `waveformNormalized` will also be all zeros.
		`revertNormalization` will divide by `amplitudeNorm` rather than by the waveform peak.

	See Also
	--------
	`normalizeArrayWaveforms`
		Normalize multiple waveforms in an array to a specified peak amplitude.

	Amplitude Scaling
	-----------------
	`normalizeWaveform` computes the absolute peak of `waveform` as the maximum of `waveform.max()`
	and `-waveform.min()`, then multiplies every sample by `amplitudeNorm / peakAbsolute`.
	`revertNormalization` reverses this by dividing every sample by the same factor.

	Examples
	--------
	Normalize a waveform and revert the normalization:

	```python
		from hunterHearsPy import normalizeWaveform

		waveformNormalized, revertNormalization = normalizeWaveform(waveform.copy())
		waveformReverted = revertNormalization(waveformNormalized)
	```

	References
	----------
	[1] `hunterHearsPy.theTypes.Waveform`

	[2] `hunterHearsPy.theTypes.NormalizationReverter`

	[3] numpy.finfo - NumPy reference
		https://numpy.org/doc/stable/reference/generated/numpy.finfo.html

	"""
	# TODO replace `numpy_finfo`.
	amplitudeNorm = amplitudeNorm or float(numpy_finfo(waveform.dtype.str).tiny.astype(waveform.dtype))

	peakAbsolute: float = abs(float(numpy_max([waveform.max(), -waveform.min()]))) or 1.0
	amplitudeAdjustment: float = amplitudeNorm / peakAbsolute

	multiply(waveform, amplitudeAdjustment, out=waveform)

	@overload
	def revertNormalization(waveformDescendant: Waveform) -> Waveform: ...
	@overload
	def revertNormalization(waveformDescendant: ArrayWaveforms) -> ArrayWaveforms: ...
	def revertNormalization(waveformDescendant: ArrayWaveforms | Waveform) -> ArrayWaveforms | Waveform:
		return divide(waveformDescendant, amplitudeAdjustment, out=waveformDescendant)

	return waveform, revertNormalization

def normalizeArrayWaveforms(arrayWaveforms: ArrayWaveforms, amplitudeNorm: float = 1.0) -> tuple[ArrayWaveforms, tuple[NormalizationReverter, ...]]:
	"""Normalize multiple waveforms in an array to a specified peak amplitude.

	(AI generated docstring)

	You can use this function to scale each `Waveform` [1] in an `ArrayWaveforms` [2] so that
	each waveform's absolute peak value equals `amplitudeNorm`. This function also returns
	`listRevertNormalization`, a list of `NormalizationReverter` [3] callables, one per waveform,
	that each reverse the scaling for the corresponding waveform at the matching last-axis index.

	`normalizeArrayWaveforms` delegates each individual waveform normalization to
	`normalizeWaveform` [4] and modifies `arrayWaveforms` in place before returning it.

	Parameters
	----------
	arrayWaveforms : ArrayWaveforms
		Array containing multiple waveforms indexed on the last axis. Shape is
		(channels, samples, waveforms).
	amplitudeNorm : float = 1.0
		Target peak amplitude. The absolute maximum value of each normalized waveform equals
		`amplitudeNorm`.

	Returns
	-------
	arrayWaveformsNormalized : ArrayWaveforms
		The array of normalized waveforms, identical to `arrayWaveforms` modified in place.
		Each waveform is scaled to peak amplitude `amplitudeNorm`.
	listRevertNormalization : tuple[NormalizationReverter, ...]
		A tuple of callables indexed in the same order as the last axis of `arrayWaveforms`.
		Each callable reverses the normalization scaling for the corresponding waveform.

	See Also
	--------
	`normalizeWaveform`
		Normalize a single waveform to a specified peak amplitude.

	Examples
	--------
	Normalize all waveforms in an array and revert each one:

	```python
		from hunterHearsPy import normalizeArrayWaveforms

		arrayNormalized, listRevertNormalization = normalizeArrayWaveforms(arrayWaveforms.copy())
		for indexWaveform in range(arrayNormalized.shape[-1]):
			arrayReverted[..., indexWaveform] = listRevertNormalization[indexWaveform](arrayReverted[..., indexWaveform])
	```

	References
	----------
	[1] `hunterHearsPy.theTypes.Waveform`

	[2] `hunterHearsPy.theTypes.ArrayWaveforms`

	[3] `hunterHearsPy.theTypes.NormalizationReverter`

	[4] `normalizeWaveform`

	"""
	listRevertNormalization: list[NormalizationReverter] = [lambda makeTypeCheckerHappy: makeTypeCheckerHappy] * arrayWaveforms.shape[-1]
	for index in range(arrayWaveforms.shape[-1]):
		arrayWaveforms[..., index], listRevertNormalization[index] = normalizeWaveform(arrayWaveforms[..., index], amplitudeNorm)
	return arrayWaveforms, tuple(listRevertNormalization)
