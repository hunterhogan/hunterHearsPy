# pyright: reportArgumentType=false
# pyright: reportUnknownVariableType=false
# ty:ignore[invalid-argument-type]
from __future__ import annotations

from typing import overload, TYPE_CHECKING
import resampy

if TYPE_CHECKING:
	from hunterHearsPy import 形floating, 形Shape
	from numpy import dtype, float32, integer, ndarray
	from typing import Any

# TODO update typeshed `resampy`.
@overload
def resampleWaveform(waveform: ndarray[形Shape, dtype[integer[Any]]], sampleRateDesired: float, sampleRateSource: float, axisTime: int = -1) -> ndarray[形Shape, dtype[float32]]: ...
@overload
def resampleWaveform(waveform: ndarray[形Shape, dtype[形floating]], sampleRateDesired: float, sampleRateSource: float, axisTime: int = -1) -> ndarray[形Shape, dtype[形floating]]: ...
def resampleWaveform(waveform: ndarray[形Shape, dtype[形floating]] | ndarray[形Shape, dtype[integer[Any]]], sampleRateDesired: float, sampleRateSource: float, axisTime: int = -1) -> ndarray[形Shape, dtype[形floating]] | ndarray[形Shape, dtype[float32]]:
	"""Resample `waveform` array to `sampleRateDesired` along the `axisTime` axis.

	Warning
	-------
	The returned `ndarray` always has a floating-point `dtype`, even if the sample rate is unchanged.

	This function is _not_ regulated by the universal settings.

	Parameters
	----------
	waveform : ndarray[形Shape, dtype[形floating] | dtype[integer[Any]]]
		Input audio data as a NumPy `ndarray`.
	sampleRateDesired : float
		Target sample rate in Hz.
	sampleRateSource : float
		Original sample rate of `waveform` in Hz.
	axisTime : int = -1
		Axis along which resampling is performed. Negative values index from the last axis.

	Returns
	-------
	waveformResampled : ndarray[形Shape, dtype[形floating] | dtype[float32]]
		Waveform sampled at `sampleRateDesired` Hz.
	"""
	return resampy.resample(waveform, sampleRateSource, sampleRateDesired, axis=axisTime)
