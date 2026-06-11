from __future__ import annotations

from typing import TYPE_CHECKING
import resampy

if TYPE_CHECKING:
	from hunterHearsPy import ArrayTypeVariable

def resampleWaveform(waveform: ArrayTypeVariable, sampleRateDesired: float, sampleRateSource: float, axisTime: int = -1) -> ArrayTypeVariable:
	"""Resample a waveform array to a target sample rate along the `axisTime` axis.

	Parameters
	----------
	waveform : ArrayTypeVariable
		Input audio data as any floating-point NumPy array.
	sampleRateDesired : float
		Target sample rate in Hz.
	sampleRateSource : float
		Original sample rate of `waveform` in Hz.
	axisTime : int = -1
		Axis along which resampling is performed. Negative values index from the last axis.

	Returns
	-------
	waveformResampled : ArrayTypeVariable
		Waveform resampled to `sampleRateDesired`.
	"""
	return resampy.resample(waveform, sampleRateSource, sampleRateDesired, axis=axisTime)
