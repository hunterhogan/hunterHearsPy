from __future__ import annotations

from humpy_cytoolz.dicttoolz import keyfilter, merge
from hunterMakesPy.dataStructures import stringItUp
from tests import pathDataSamples, pathDataSamplesExpected
from typing import TYPE_CHECKING
import inspect
import numpy
import pytest

if TYPE_CHECKING:
	from hunterHearsPy import 个, 形ndarray
	from pathlib import Path
	from torch.types import Device
	from typing import Any

#================== Settings =====================================================================

@pytest.fixture()
def approx_abs(request: pytest.FixtureRequest) -> float:
	"""Return the absolute tolerance for approximate comparisons."""
	return 1e-12

@pytest.fixture()
def approx_rel(request: pytest.FixtureRequest) -> float:
	"""Return the relative tolerance for approximate comparisons."""
	return 1e-6

@pytest.fixture()
def atol(request: pytest.FixtureRequest) -> float:
	"""Return the absolute tolerance for `numpy.allclose` comparisons."""
	return 1e-08

@pytest.fixture()
def rtol(request: pytest.FixtureRequest) -> float:
	"""Return the relative tolerance for `numpy.allclose` comparisons."""
	return 1e-05

#================== Assert ========================================================================

def assert_allclose(actual: Any, expected: Any, rtol: float, atol: float, function: str, *arguments: Any, **keywordArguments: Any) -> None:
	assert numpy.allclose(actual, expected, rtol, atol), messageTestFailure(actual, expected, function, *arguments, **keywordArguments)

def assert_approx(
	actual: 个, expected: 个, pytest_rel: float, pytest_abs: float, function: str, *arguments: Any, **keywordArguments: Any
) -> None:
	assert actual == pytest.approx(expected, pytest_rel, pytest_abs, nan_ok=True), messageTestFailure(
		actual, expected, function, *arguments, **keywordArguments
	)

def assert_array_equal(actual: 形ndarray, expected: 形ndarray, function: str, *arguments: Any, **keywordArguments: Any) -> None:
	"""Assert that two arrays are equal, and if not, raise an AssertionError with a detailed message."""
	assert numpy.array_equal(actual, expected), messageTestFailure_ndarray(actual, expected, function, *arguments, **keywordArguments)

def assertEqualTo(actual: 个, expected: 个, function: str, *arguments: Any, **keywordArguments: Any) -> None:
	"""Assert that two arrays are equal, and if not, raise an AssertionError with a detailed message."""
	assert actual == expected, messageTestFailure(actual, expected, function, *arguments, **keywordArguments)

#------------------ Messages ------------------------------------------------------------------------------

def messageTestFailure(actual: Any, expected: Any, function: str, *arguments: Any, **keywordArguments: Any) -> str:
	"""Format assertion message for any test comparison."""
	parameters: list[str] = list(map(repr, arguments))
	parameters.extend(f'{keyAndValue[0]}={keyAndValue[1]!r}' for keyAndValue in keywordArguments.items())
	return f'{function}({", ".join(parameters)}) = {actual!r}, but {expected = }.'

def messageTestFailure_ndarray(actual: 形ndarray, expected: 形ndarray, function: str, *arguments: Any, **keywordArguments: Any) -> str:
	parameters: list[str] = list(map(repr, arguments))
	parameters.extend(f'{keyAndValue[0]}={keyAndValue[1]!r}' for keyAndValue in keywordArguments.items())
	return f'{function}({", ".join(parameters)}) = {actual.shape=},\t{actual.dtype=}, but {expected.shape=}, {expected.dtype=}.'

#================== Parameters ========================================================================

@pytest.fixture(params=tuple(map(pytest.param, (None, 'cpu'))))
def device(request: pytest.FixtureRequest) -> Device | None:
	return request.param

@pytest.fixture()
def expected(request: pytest.FixtureRequest) -> 形ndarray:
	filenameStemExpected: str = '__'.join((request.function.__name__
		, *(f'{keyAndValue[0]}~{"".join(stringItUp(keyAndValue[1]) or ["None"])}'
			for keyAndValue in keyfilter(request.node.callspec.params.keys().__contains__  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportUnknownArgumentType]
				, merge(inspect.signature(request.function).parameters, request.node.callspec.params)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
			).items()
	)))
	pathFilenameExpected: Path = pathDataSamplesExpected / f'{filenameStemExpected}.npy'
	return numpy.load(pathFilenameExpected, mmap_mode='r', allow_pickle=False)

@pytest.fixture()
def pathFilename(request: pytest.FixtureRequest) -> Path:
	filename: str = request.param
	return pathDataSamples / filename
