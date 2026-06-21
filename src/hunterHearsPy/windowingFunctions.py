"""Generate windowing functions for signal processing.

Contents
--------
Functions
	cosineWings
		Generate a cosine-tapered windowing function with a flat center and tapered ends.
	equalPower
		Generate an equal-power taper for crossfades.
	halfsine
		Generate a half-sine windowing function.
	tukey
		Generate a Tukey windowing function with optional taper control.
"""
from __future__ import annotations

from numpy import cos, pi, sin
from typing import TYPE_CHECKING
import numpy
import scipy.signal.windows as SciPy

if TYPE_CHECKING:
	from hunterHearsPy import WindowingFunction

def _getLengthTaper(lengthSupport: int, ratioTaper: float) -> int:
	"""I use this to compute and validate the taper length for a windowingFunction.

	Parameters
	----------
	lengthSupport : int
		Total length of the windowingFunction in samples.
	ratioTaper : float
		Fraction of the total windowingFunction width to allocate to tapering, in the closed interval
		[0, 1].

	Returns
	-------
	lengthTaper : int
		Number of samples to use for each tapered end.

	Raises
	------
	ValueError
		If `ratioTaper` is outside the closed interval [0, 1].
	"""
	if 0 <= ratioTaper <= 1:
		lengthTaper = int(lengthSupport * ratioTaper / 2)
	else:
		message: str = f"I received `{ratioTaper = }`. If set, `ratioTaper` must be between 0 and 1, inclusive."
		raise ValueError(message)
	return lengthTaper

def cosineWings(lengthSupport: int, ratioTaper: float = 0.1) -> WindowingFunction:
	"""Generate a cosine-tapered windowingFunction with a flat center and tapered ends.

	You can use this function to produce a 1-D windowingFunction of length `lengthSupport` whose ends
	are tapered with a cosine shape while the center remains at unity. The taper occupies `ratioTaper`
	of the total windowingFunction width and is split equally between the two ends.

	Parameters
	----------
	lengthSupport : int
		Total length of the windowingFunction in samples.
	ratioTaper : float = 0.1
		Fraction of the total windowingFunction to taper. Must be in the closed interval [0, 1]. The
		computed taper length per end is `int(lengthSupport * ratioTaper / 2)`.

	Returns
	-------
	windowingFunction : WindowingFunction
		1-D array of shape `(lengthSupport,)` containing values in [0, 1]. The centre region is 1.0
		and each end contains a cosine-shaped ramp from 0 → 1 (or 1 → 0) of length `lengthTaper`.
	"""
	lengthTaper: int = _getLengthTaper(lengthSupport, ratioTaper)

	windowingFunction: WindowingFunction = numpy.ones(shape=lengthSupport)
	if 0 < lengthTaper:
		taper = 1 - cos(numpy.linspace(start=0, stop=pi / 2, num=lengthTaper, dtype=windowingFunction.dtype))
		windowingFunction[0:lengthTaper] = taper
		windowingFunction[-lengthTaper:None] = taper[::-1]
	return windowingFunction

def equalPower(lengthSupport: int, ratioTaper: float = 0.1) -> WindowingFunction:
	"""Generate an equal-power windowingFunction suitable for crossfades.

	You can use this function to build a 1-D windowingFunction whose ends follow a square-root ramp.
	This produces an equal-power fade for crossfades, where amplitude ramps follow √(linear) shapes to
	preserve perceived loudness during mixing.

	Parameters
	----------
	lengthSupport : int
		Total length of the windowingFunction in samples.
	ratioTaper : float = 0.1
		Fraction of the total windowingFunction to taper. Must be in the closed interval [0, 1]. The
		computed taper length per end is `int(lengthSupport * ratioTaper / 2)`.

	Returns
	-------
	windowingFunction : WindowingFunction
		1-D array of shape `(lengthSupport,)` containing values in [0, 1]. The central region is 1.0
		and each end contains a √-shaped ramp of length `lengthTaper`.
	"""
	lengthTaper: int = _getLengthTaper(lengthSupport, ratioTaper)

	windowingFunction: WindowingFunction = numpy.ones(shape=lengthSupport)
	if 0 < lengthTaper:
		taper = numpy.sqrt(numpy.linspace(start=0, stop=1, num=lengthTaper, dtype=windowingFunction.dtype))
		windowingFunction[0:lengthTaper] = taper
		windowingFunction[-lengthTaper:None] = taper[::-1]
	return windowingFunction

def halfsine(lengthSupport: int) -> WindowingFunction:
	"""Generate a half-sine windowingFunction of the requested length.

	This function returns a 1-D half-sine windowingFunction of length `lengthSupport`. The value at
	sample index `n` is `sin(π * (n + 0.5) / lengthSupport)`, producing a smoothly varying
	windowingFunction that starts and ends away from zero, commonly used in short-time analysis and
	overlap-add reconstruction.

	Parameters
	----------
	lengthSupport : int
		Total length of the windowingFunction in samples.

	Returns
	-------
	windowingFunction : WindowingFunction
		1-D array of shape `(lengthSupport,)` containing the half-sine values.

	References
	----------
	[1] Short-Time Fourier Transform and Its Inverse.
	https://eeweb.engineering.nyu.edu/iselesni/EL713/STFT/stft_inverse.pdf
	"""
	return sin(pi * (numpy.arange(lengthSupport) + 0.5) / lengthSupport, dtype=numpy.float64)

def tukey(lengthSupport: int, ratioTaper: float = 0.1, **keywordArguments: float) -> WindowingFunction:
	"""Generate a Tukey windowingFunction.

	You can use this function to obtain a tapered windowingFunction where the central region is
	constant and the ends are cosine-shaped. By default, the function uses `ratioTaper` as the Tukey
	`alpha` value. If an explicit `alpha` is provided via keyword arguments, that value is used
	instead of `ratioTaper`.

	Parameters
	----------
	lengthSupport : int
		Total length of the windowingFunction in samples.
	ratioTaper : float = 0.1
		Default fraction of the windowingFunction to taper. Provided for API symmetry with other
		windowingFunction constructors.

	Other Parameters
	----------------
	alpha : float = ratioTaper
		Equivalent to SciPy's `alpha` parameter. If provided, this overrides `ratioTaper`. Values are
		typically in the closed interval [0, 1].

	Returns
	-------
	windowingFunction : WindowingFunction
		1-D array of shape `(lengthSupport,)`.
	"""
	alpha: float = keywordArguments.get('alpha', ratioTaper)  # Are you tempted to use `or 0.1`? Don't be: it will override the user's value for `ratioTaper=0`.
	return SciPy.tukey(lengthSupport, alpha)
