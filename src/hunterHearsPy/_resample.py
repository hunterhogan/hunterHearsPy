from __future__ import annotations

from typing import TYPE_CHECKING
import resampy

if TYPE_CHECKING:
	from hunterHearsPy import ShapeTypeVariable
	from numpy import dtype, floating, ndarray
	from typing import Any

# TODO? Test `resampy` to see if it will accept integer waveforms and if it will return an integer
# array, especially if "If sr_new == sr_orig, then a copy of x is returned with no interpolation
# performed." Then update typeshed if necessary.
def resampleWaveform(waveform: ndarray[ShapeTypeVariable, dtype[floating[Any]]], sampleRateDesired: float, sampleRateSource: float, axisTime: int = -1) -> ndarray[ShapeTypeVariable, dtype[floating[Any]]]:
	"""Resample `waveform` array to `sampleRateDesired` along the `axisTime` axis.

	This function is _not_ regulated by the universal settings.

	Parameters
	----------
	waveform : ndarray[ShapeTypeVariable, dtype[floating[Any]]]
		Input audio data as a NumPy array.
	sampleRateDesired : float
		Target sample rate in Hz.
	sampleRateSource : float
		Original sample rate of `waveform` in Hz.
	axisTime : int = -1
		Axis along which resampling is performed. Negative values index from the last axis.

	Returns
	-------
	waveformResampled : ndarray[ShapeTypeVariable, dtype[floating[Any]]]
		Waveform resampled to `sampleRateDesired`.
	"""
	return resampy.resample(waveform, sampleRateSource, sampleRateDesired, axis=axisTime)
