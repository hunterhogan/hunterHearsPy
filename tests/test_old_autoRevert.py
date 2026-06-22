from __future__ import annotations

from hunterHearsPy import moveToAxisOfOperation
from numpy import int64
from tests import messageTestFailure
from typing import Any, TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from numpy.typing import NDArray

@pytest.fixture
def arrayAxisOperation() -> NDArray[int64]:
	"""You can use this fixture to test axis movement with deterministic integer data."""
	return ((numpy.arange(2 * 3 * 5 * 7, dtype=int64) + 5) * 3).reshape((2, 3, 5, 7))

@pytest.mark.parametrize('axisSource, axisOfOperation', [(0, -1), (1, -1), (2, 0), (-1, 1)])
def test_moveToAxisOfOperation_movesAxisAndPreservesOriginalArray(
	arrayAxisOperation: NDArray[numpy.int64], axisSource: int, axisOfOperation: int
) -> None:
	arrayOriginal: NDArray[numpy.int64] = arrayAxisOperation.copy()
	shapeOriginal: tuple[Any, ...] = arrayAxisOperation.shape
	arrayExpectedMoved: NDArray[numpy.int64] = numpy.moveaxis(arrayOriginal, axisSource, axisOfOperation)
	valueOffset: int = 13

	with moveToAxisOfOperation(arrayAxisOperation, axisSource, axisOfOperation) as arrayStandardized:
		assert arrayStandardized.shape == arrayExpectedMoved.shape, messageTestFailure(
			arrayStandardized.shape,
			arrayExpectedMoved.shape,
			moveToAxisOfOperation.__name__,
			axisSource=axisSource,
			axisOfOperation=axisOfOperation,
		)
		assert numpy.array_equal(arrayStandardized, arrayExpectedMoved), messageTestFailure(
			arrayStandardized, arrayExpectedMoved, moveToAxisOfOperation.__name__, axisSource=axisSource, axisOfOperation=axisOfOperation
		)

		arrayStandardized += valueOffset

	assert arrayAxisOperation.shape == shapeOriginal, messageTestFailure(
		arrayAxisOperation.shape, shapeOriginal, moveToAxisOfOperation.__name__, axisSource=axisSource, axisOfOperation=axisOfOperation
	)
	assert numpy.array_equal(arrayAxisOperation, arrayOriginal + valueOffset), messageTestFailure(
		arrayAxisOperation, arrayOriginal + valueOffset, moveToAxisOfOperation.__name__, axisSource=axisSource, axisOfOperation=axisOfOperation
	)
