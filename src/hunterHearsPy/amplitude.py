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
from numpy import divide, finfo as numpy_finfo, float32, iinfo as numpy_iinfo, max as numpy_max, multiply, ndarray
from soundfile import dtype_str
from typing import Any, overload, TYPE_CHECKING
from typing_extensions import TypeVar
import numpy

if TYPE_CHECKING:
	from hunterHearsPy import ArrayWaveforms, NormalizationReverter, Waveform
	from numpy import dtype, floating, integer
	from pathlib import PurePath

Axes = TypeVar('Axes', tuple[int, int], tuple[int, int, int])

def amplitudeIntegerToFloating(arrayTarget: ndarray[Axes, dtype[integer[Any]]]) -> ndarray[Axes, dtype[floating[Any]]]:
	integerInformation: numpy_iinfo[integer] = numpy_iinfo(arrayTarget.dtype.str)
	dtypeFloating: dtype[floating[Any]] = numpy.promote_types(arrayTarget.dtype, float32)
	arrayFloating: ndarray[Axes, dtype[floating[Any]]] = numpy.astype(arrayTarget, dtypeFloating, copy=False)
	if integerInformation.min < 0:
		arrayFloating /= -integerInformation.min
	else:
		arrayFloating -= (integerInformation.max + zeroIndexed) / 2
		arrayFloating /= (integerInformation.max + zeroIndexed) / 2
	return arrayFloating

def amplitudeToSoundfile(arrayTarget: ndarray[Axes, dtype[Any]]) -> ndarray[Axes, dtype[Any]]:
	dtypeSoundfile: tuple[dtype[Any], ...] = tuple(map(numpy.dtype, dtype_str.__args__))
	dtypeSoundfileFloating: tuple[dtype[Any], ...] = tuple(
		filter(lambda dtypeCandidate: numpy.issubdtype(dtypeCandidate, numpy.floating), dtypeSoundfile)
	)
	dtypeSoundfileInteger: tuple[dtype[Any], ...] = tuple(
		filter(lambda dtypeCandidate: numpy.issubdtype(dtypeCandidate, numpy.integer), dtypeSoundfile)
	)
	dtypeFloatingMaximum: dtype[Any] = max(dtypeSoundfileFloating, key=lambda dtypeCandidate: dtypeCandidate.itemsize)
	dtypeFloatingTarget: dtype[Any] = min(
		filter(lambda dtypeCandidate: arrayTarget.dtype.itemsize <= dtypeCandidate.itemsize, dtypeSoundfileFloating)
		, key=lambda dtypeCandidate: dtypeCandidate.itemsize
		, default=dtypeFloatingMaximum
	)
	dtypeIntegerMaximum: dtype[Any] = max(dtypeSoundfileInteger, key=lambda dtypeCandidate: dtypeCandidate.itemsize)
	dtypeIntegerTarget: dtype[Any] = min(
		filter(lambda dtypeCandidate: arrayTarget.dtype.itemsize <= dtypeCandidate.itemsize, dtypeSoundfileInteger)
		, key=lambda dtypeCandidate: dtypeCandidate.itemsize
		, default=dtypeIntegerMaximum
	)

	if arrayTarget.dtype.name in dtype_str.__args__:
		arraySoundfile = arrayTarget
	elif numpy.issubdtype(arrayTarget.dtype, numpy.floating):
		arraySoundfile = numpy.astype(arrayTarget, dtypeFloatingTarget, copy=False)
	elif numpy.issubdtype(arrayTarget.dtype, numpy.integer):
		integerInformationSource: numpy_iinfo[integer] = numpy_iinfo(arrayTarget.dtype)
		integerInformationTarget: numpy_iinfo[integer] = numpy_iinfo(dtypeIntegerTarget)
		dtypeFloating: dtype[floating[Any]] = numpy.promote_types(dtypeIntegerTarget, float32)
		arraySoundfile = numpy.astype(arrayTarget, dtypeFloating, copy=False)
		if integerInformationSource.min < 0:
			arraySoundfile /= -integerInformationSource.min
		else:
			amplitudeMiddle: float = (integerInformationSource.max + zeroIndexed) / 2
			arraySoundfile -= amplitudeMiddle
			arraySoundfile /= amplitudeMiddle
		if integerInformationTarget.min < 0:
			arraySoundfile *= -integerInformationTarget.min
		else:
			amplitudeMiddle = (integerInformationTarget.max + zeroIndexed) / 2
			arraySoundfile *= amplitudeMiddle
			arraySoundfile += amplitudeMiddle
		numpy.rint(arraySoundfile, out=arraySoundfile)
		numpy.clip(arraySoundfile, integerInformationTarget.min, integerInformationTarget.max, out=arraySoundfile)
		arraySoundfile = numpy.astype(arraySoundfile, dtypeIntegerTarget, copy=False)
	else:
		try:
			arraySoundfile = numpy.astype(arrayTarget, dtypeFloatingMaximum, copy=False)
		except (TypeError, ValueError) as error:
			from hunterHearsPy._io import saveOnError  # noqa: PLC0415

			pathFilename: PurePath = saveOnError(arrayTarget)
			message: str = (
				'I could not convert `arrayTarget` to a soundfile-compatible dtype. '
				"I saved `arrayTarget` to a file in this computer's temporary directory so you might recover the data. "
				f'{arrayTarget.shape = }, {arrayTarget.dtype = }\n'
				f'{pathFilename = }'
			)
			raise TypeError(message) from error
	return arraySoundfile

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

def normalizeArrayWaveforms(
	arrayWaveforms: ArrayWaveforms, amplitudeNorm: float = 1.0
) -> tuple[ArrayWaveforms, list[NormalizationReverter]]:
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
	listRevertNormalization : list[NormalizationReverter]
		A list of callables indexed in the same order as the last axis of `arrayWaveforms`.
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
	return arrayWaveforms, listRevertNormalization
