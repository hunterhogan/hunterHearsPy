# pyright: reportArgumentType=false
# pyright: reportUnknownVariableType=false
# ty:ignore[invalid-argument-type]
from __future__ import annotations

from typing import overload, TYPE_CHECKING
import resampy

if TYPE_CHECKING:
	from hunterHearsPy import Floater, ShapeTypeVariable
	from numpy import dtype, float32, integer, ndarray
	from typing import Any

# TODO update typeshed `resampy`.
@overload
def resampleWaveform(waveform: ndarray[ShapeTypeVariable, dtype[integer[Any]]], sampleRateDesired: float, sampleRateSource: float, axisTime: int = -1) -> ndarray[ShapeTypeVariable, dtype[float32]]: ...
@overload
def resampleWaveform(waveform: ndarray[ShapeTypeVariable, dtype[Floater]], sampleRateDesired: float, sampleRateSource: float, axisTime: int = -1) -> ndarray[ShapeTypeVariable, dtype[Floater]]: ...
def resampleWaveform(waveform: ndarray[ShapeTypeVariable, dtype[Floater]] | ndarray[ShapeTypeVariable, dtype[integer[Any]]], sampleRateDesired: float, sampleRateSource: float, axisTime: int = -1) -> ndarray[ShapeTypeVariable, dtype[Floater]] | ndarray[ShapeTypeVariable, dtype[float32]]:
	"""Resample `waveform` array to `sampleRateDesired` along the `axisTime` axis.

	This function is _not_ regulated by the universal settings.

	Parameters
	----------
	waveform : ndarray[ShapeTypeVariable, dtype[Floater] | dtype[integer[Any]]]
		Input audio data as a NumPy array.
	sampleRateDesired : float
		Target sample rate in Hz.
	sampleRateSource : float
		Original sample rate of `waveform` in Hz.
	axisTime : int = -1
		Axis along which resampling is performed. Negative values index from the last axis.

	Returns
	-------
	waveformResampled : ndarray[ShapeTypeVariable, dtype[Floater] | dtype[float32]]
		Waveform resampled to `sampleRateDesired`.
	"""
	return resampy.resample(waveform, sampleRateSource, sampleRateDesired, axis=axisTime)
