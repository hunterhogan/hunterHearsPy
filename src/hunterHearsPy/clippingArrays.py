"""Clip and limit array values using magnitude-based hard limits.

(AI generated docstring)

You can use this module to apply hard clipping [1] to NumPy [2] arrays, constraining element
magnitudes within bounds defined by a comparand array. The module is not yet fully implemented
and may not correctly handle all cases.

Contents
--------
Functions
	applyHardLimit
		Clip the elements of a real-valued array to stay within the magnitude of a comparand.
	applyHardLimitComplexValued
		Clip the elements of a complex-valued array using magnitude-based scaling.

References
----------
[1] Clipping (signal processing) - Wikipedia
	https://en.wikipedia.org/wiki/Clipping_(signal_processing)

[2] NumPy
	https://numpy.org/doc/stable/
"""
from __future__ import annotations

from numpy import absolute, float64, multiply, ones_like
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from hunterHearsPy import 形ndarray, 形Shape
	from numpy import complexfloating, dtype, floating, ndarray
	from numpy.typing import ArrayLike
	from typing import Any

# NOTE I wish I had written down my evidence for why `applyHardLimit` is superior to `numpy.clip`.
# Nevertheless, I'm confident I needed this function. Use an array for `comparand` to get the full
# value of this function.

def applyHardLimit(arrayTarget: 形ndarray, comparand: ArrayLike = 1.0) -> 形ndarray:
	"""Clip the elements of `arrayTarget` to the magnitude bounds defined by `comparand`.

	This function applies a hard amplitude limit element-wise to `arrayTarget`. Elements whose
	magnitude exceeds the corresponding magnitude of `comparand` are reduced toward zero until
	the element magnitude equals the comparand magnitude. The operation modifies `arrayTarget`
	in place and returns a reference to it.

	Parameters
	----------
	arrayTarget : 形ndarray
		The array to clip. Modified in place.
	comparand : ArrayLike = 1.0
		The magnitude threshold. Elements of `arrayTarget` whose magnitude strictly exceeds the
		corresponding magnitude in `comparand` are clipped to that comparand magnitude.

	Returns
	-------
	arrayClipped : 形ndarray
		A reference to the modified `arrayTarget`.

	See Also
	--------
	`applyHardLimitComplexValued`
		Clip complex-valued array elements using magnitude-based scaling.

	References
	----------
	[1] Clipping (signal processing) - Wikipedia
		https://en.wikipedia.org/wiki/Clipping_(signal_processing)

	[2] numpy.typing.NDArray
		https://numpy.org/doc/stable/reference/typing.html#numpy.typing.NDArray

	[3] numpy.typing.ArrayLike
		https://numpy.org/doc/stable/reference/typing.html#numpy.typing.ArrayLike

	"""
	selectAboveThreshold = absolute(comparand) - absolute(arrayTarget) < 0.0
	reduction = arrayTarget - (absolute(arrayTarget) - absolute(comparand))
	arrayTarget[selectAboveThreshold] = reduction[selectAboveThreshold]
	return arrayTarget

def applyHardLimitComplexValued(
		arrayTarget: ndarray[形Shape, dtype[complexfloating[Any, Any]]]
		, comparand: ndarray[形Shape, dtype[floating[Any] | complexfloating[Any, Any]]]
		, penalty: float = 1.0
	) -> ndarray[形Shape, dtype[complexfloating[Any, Any]]]:
	"""Clip, with `penalty`, complex-valued `arrayTarget` exceeding `comparand` magnitudes.

	This function applies a magnitude-based hard limit to each element of `arrayTarget`. When the
	magnitude of an element strictly exceeds the corresponding value in `comparand`, the element is
	scaled down by a power of the ratio of comparand magnitude to target magnitude. Elements whose
	magnitudes are within the limit are left unchanged. This function returns a new array and does
	not modify `arrayTarget` in place.

	Parameters
	----------
	arrayTarget : NDArray[complexfloating[Any, Any]]
		The complex-valued array to clip.
	comparand : NDArray[floating[Any] | complexfloating[Any, Any]]
		The magnitude threshold array. Only the magnitudes of `comparand` values are used.
	penalty : float = 1.0
		Exponent applied to the scaling factor when limiting is needed. Values greater than 1.0
		produce more aggressive clipping; values between 0.0 and 1.0 produce less aggressive clipping.

	Returns
	-------
	arrayResult : NDArray[complexfloating[Any, Any]]
		A new array with the same shape and dtype as `arrayTarget`, with element magnitudes
		clipped according to `comparand`.

	See Also
	--------
	`applyHardLimit`
		Clip real-valued array elements to stay within the magnitude of a comparand.

	Mathematics
	-----------
	magnitude scaling : equation
	```
		Let a ≜ `arrayTarget`,  c ≜ `comparand`,  p ≜ `penalty`,
			s ≜ (|cᵢ| / |aᵢ|)

		For each element i where |aᵢ| > |cᵢ|:
			resultᵢ = aᵢ × sᵖ

		For each element i where |aᵢ| ≤ |cᵢ|:
			resultᵢ = aᵢ
	```

	References
	----------
	[1] Clipping (signal processing) - Wikipedia
		https://en.wikipedia.org/wiki/Clipping_(signal_processing)

	[2] numpy.typing.NDArray
		https://numpy.org/doc/stable/reference/typing.html#numpy.typing.NDArray

	"""
	arrayTargetMagnitude: ndarray[形Shape, dtype[float64]] = absolute(arrayTarget, dtype=float64)  # pyright: ignore[reportAssignmentType] # ty:ignore[invalid-assignment]

	comparandMagnitude: ndarray[形Shape, dtype[float64]] = absolute(comparand, dtype=float64)  # pyright: ignore[reportAssignmentType] # ty:ignore[invalid-assignment]

	selectAboveThreshold: ndarray[形Shape, dtype[float64]] = ((comparandMagnitude - arrayTargetMagnitude) < 0.0).astype(float64)  # pyright: ignore[reportAssignmentType] # ty:ignore[invalid-assignment]

	arrayClippingCoefficientsMagnitude: ndarray[形Shape, dtype[float64]] = comparandMagnitude[selectAboveThreshold] / arrayTargetMagnitude[selectAboveThreshold]  # pyright: ignore[reportArgumentType, reportCallIssue, reportUnknownVariableType]
	# TODO I don't remember why I created complexfloating `arrayClippingCoefficients` instead of just
	# using `arrayClippingCoefficientsMagnitude`. I made this long enough ago that I did it because I
	# was a n00b. Oh, I see. `arrayClippingCoefficientsMagnitude` is empty in cells
	# ~selectAboveThreshold. I probably didn't understand that at the time, so `ones_like` fixed the
	# problem.
	# TODO I need real tests.
	arrayClippingCoefficients = ones_like(arrayTarget, dtype=arrayTarget.dtype)
	arrayClippingCoefficients[selectAboveThreshold] = arrayClippingCoefficientsMagnitude**penalty  # pyright: ignore[reportArgumentType, reportCallIssue]

	return multiply(arrayTarget, arrayClippingCoefficients)  # pyright: ignore[reportReturnType] # ty:ignore[invalid-return-type]
