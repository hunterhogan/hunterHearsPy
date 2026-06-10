from __future__ import annotations

from typing import TYPE_CHECKING
import resampy

if TYPE_CHECKING:
	from numpy import dtype, floating, ndarray
	from typing import Any

def resampleWaveform(waveform: ndarray[tuple[int, ...], dtype[floating[Any]]], sampleRateDesired: float, sampleRateSource: float, axisTime: int = -1) -> ndarray[tuple[int, ...], dtype[floating[Any]]]:
	"""Resample a waveform array to a target sample rate using `resampy` [1].

	You can use this function to change the sample rate of any floating-point NumPy array [2].
	`resampleWaveform` passes `waveform` to `resampy.resample` [1] along the `axisTime` axis.
	When `sampleRateSource` equals `sampleRateDesired`, `resampleWaveform` returns `waveform`
	unchanged without invoking `resampy`.

	Parameters
	----------
	waveform : ndarray[tuple[int, ...], dtype[floating[Any]]]
		Input audio data as any floating-point NumPy array [2].
	sampleRateDesired : float
		Target sample rate in Hz.
	sampleRateSource : float
		Original sample rate of `waveform` in Hz.
	axisTime : int = -1
		Axis along which resampling is performed. Negative values index from the last axis.

	Returns
	-------
	waveformResampled : ndarray[tuple[int, ...], dtype[floating[Any]]]
		Waveform resampled to `sampleRateDesired`. Returns `waveform` unchanged when
		`sampleRateSource` equals `sampleRateDesired`.

	Sample Rate Rounding
	--------------------
	Both `sampleRateDesired` and `sampleRateSource` are rounded to the nearest integer
	before passing to `resampy.resample` [1]. `resampy` expects integer sample rates.

	References
	----------
	[1] resampy — efficient signal resampling
		https://resampy.readthedocs.io/en/stable/

	[2] numpy.ndarray
		https://numpy.org/doc/stable/reference/index.html

	"""
	if sampleRateSource != sampleRateDesired:
		sampleRateDesired = round(sampleRateDesired)
		sampleRateSource = round(sampleRateSource)
		waveformResampled: ndarray[tuple[int, ...], dtype[floating[Any]]] = resampy.resample(waveform, sampleRateSource, sampleRateDesired, axis=axisTime)
		return waveformResampled
	return waveform
