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

from numpy import divide, finfo as numpy_finfo, max as numpy_max, multiply
from typing import cast, overload, TYPE_CHECKING

if TYPE_CHECKING:
	from hunterHearsPy import ArrayWaveforms, NormalizationReverter, Waveform

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
	amplitudeNorm = amplitudeNorm or float(numpy_finfo(waveform.dtype).tiny.astype(waveform.dtype))

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

def normalizeArrayWaveforms(arrayWaveforms: ArrayWaveforms, amplitudeNorm: float = 1.0) -> tuple[ArrayWaveforms, list[NormalizationReverter]]:
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
			arrayReverted[..., indexWaveform] = listRevertNormalization[indexWaveform](
				arrayReverted[..., indexWaveform]
			)
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
		arrayWaveforms[..., index], listRevertNormalization[index] = normalizeWaveform(cast("Waveform", arrayWaveforms[..., index]), amplitudeNorm)
	return arrayWaveforms, listRevertNormalization
