# ruff: noqa: RUF069
# pyright: reportArgumentType=false
# ty:ignore[invalid-argument-type]
# ty:ignore[unresolved-attribute]
from __future__ import annotations

from hunterHearsPy import cosineWings, equalPower, halfsine, tukey
from tests.conftest import messageTestFailure, prototype_numpyAllClose, prototype_numpyArrayEqual
from typing import Any, TYPE_CHECKING
import numpy
import pytest
import scipy.signal.windows as SciPy

torch = pytest.importorskip('torch')
from hunterHearsPy import cosineWingsTensor, equalPowerTensor, halfsineTensor, tukeyTensor  # noqa: E402

if TYPE_CHECKING:
	from collections.abc import Callable
	from torch import Tensor

"""Section: Windowing function testing utilities"""

@pytest.fixture(params=[256, 1024, 1024 * 8, 44100, 44100 * 11])
def lengthWindow(request: pytest.FixtureRequest) -> int:
	return request.param

@pytest.fixture(params=[0.0, 0.1, 0.5, 1.0])
def ratioTaper(request: pytest.FixtureRequest) -> float:
	return request.param

listDevices: list[str] = ['cpu']
if torch is not None and torch.cuda.is_available():
	listDevices.append('cuda')

@pytest.fixture(params=listDevices)
def device(request: pytest.FixtureRequest) -> str:
	if torch is None:
		pytest.skip('torch is not installed')
	return request.param

@pytest.mark.parametrize('ratioTaper', [0.0, 0.1, 0.5, 1.0])
def test_cosineWingsArray(ratioTaper: float, lengthWindow: int) -> None:
	arrayWindow = cosineWings(lengthWindow, ratioTaper=ratioTaper)
	assert arrayWindow.shape == (lengthWindow,), messageTestFailure(arrayWindow.shape, (lengthWindow,), 'cosineWings shape check')
	if ratioTaper == 0.0:
		# Expect an all-ones array
		prototype_numpyArrayEqual(numpy.ones(lengthWindow), cosineWings, lengthWindow, ratioTaper=0.0)

@pytest.mark.parametrize('ratioTaper', [0.0, 0.1, 0.5, 1.0])
def test_equalPowerArray(ratioTaper: float, lengthWindow: int) -> None:
	arrayWindow = equalPower(lengthWindow, ratioTaper=ratioTaper)
	assert arrayWindow.shape == (lengthWindow,), messageTestFailure(arrayWindow.shape, (lengthWindow,), 'equalPower shape check')
	if ratioTaper == 0.0:
		# Expect an all-ones array
		prototype_numpyArrayEqual(numpy.ones(lengthWindow), equalPower, lengthWindow, ratioTaper=0.0)

def test_halfsineArray(lengthWindow: int) -> None:
	arrayWindow = halfsine(lengthWindow)
	assert arrayWindow.shape == (lengthWindow,), messageTestFailure(arrayWindow.shape, (lengthWindow,), 'halfsine shape check')
	assert numpy.all(arrayWindow >= 0), 'halfsine should yield non-negative coefficients'
	assert numpy.all(arrayWindow <= 1), 'halfsine should yield coefficients no greater than 1'

def test_halfsine_edge_value(lengthWindow: int) -> None:
	arrayWindow = halfsine(lengthWindow)
	expectedEdgeValue = numpy.sin(numpy.pi * 0.5 / lengthWindow)
	assert numpy.allclose(arrayWindow[0], expectedEdgeValue), messageTestFailure(
		arrayWindow[0], expectedEdgeValue, 'halfsine edge value'
	)

@pytest.mark.parametrize('ratioTaper', [0.0, 0.1, 0.5, 1.0])
def test_tukeyArray(ratioTaper: float, lengthWindow: int) -> None:
	arrayWindow = tukey(lengthWindow, ratioTaper=ratioTaper)
	assert arrayWindow.shape == (lengthWindow,), messageTestFailure(arrayWindow.shape, (lengthWindow,), 'tukey shape check')

def test_tukey_backward_compatibility() -> None:
	arrayExpected = tukey(10, ratioTaper=0.5)
	prototype_numpyAllClose(arrayExpected, None, None, tukey, 10, alpha=0.5)

def test_tukey_special_cases(lengthWindow: int) -> None:
	prototype_numpyArrayEqual(numpy.ones(lengthWindow), tukey, lengthWindow, ratioTaper=0.0)
	prototype_numpyAllClose(SciPy.hann(lengthWindow), None, None, tukey, lengthWindow, ratioTaper=1.0)

@pytest.mark.parametrize('functionWindowingInvalid', [cosineWings, equalPower])
def test_invalidTaperRatio(functionWindowingInvalid: Callable[..., numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]]) -> None:
	with pytest.raises(ValueError):
		functionWindowingInvalid(256, ratioTaper=-0.1)
	with pytest.raises(ValueError):
		functionWindowingInvalid(256, ratioTaper=1.1)

"""
Section: Tests for PyTorch tensor variants of windowing functions
"""

def prototype_tensorEquivalent(
	functionNdarrayOriginal: Callable[..., numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]],
	functionTensorTarget: Callable[..., Tensor],
	device: str,
	*arguments: Any,
	**keywordArguments: Any,
) -> None:
	"""
	Template for tests that verify tensor-based functions produce the same results as their numpy counterparts.
	"""
	ndarray = functionNdarrayOriginal(*arguments, **keywordArguments)
	tensor = functionTensorTarget(*arguments, device=torch.device(device), **keywordArguments)

	assert tensor.device.type == device, messageTestFailure(
		tensor.device.type, device, f'{functionTensorTarget.__name__} device check'
	)
	assert tensor.dtype == torch.float32, messageTestFailure(
		tensor.dtype, torch.float32, f'{functionTensorTarget.__name__} dtype check'
	)
	assert tensor.shape == torch.Size([ndarray.shape[0]]), messageTestFailure(
		tensor.shape, ndarray.shape, f'{functionTensorTarget.__name__} shape check'
	)

	# Convert tensor to numpy for comparison with original array
	tensorAsNumpy = tensor.cpu().numpy()
	assert numpy.allclose(ndarray, tensorAsNumpy), messageTestFailure(
		ndarray, tensorAsNumpy, f'{functionTensorTarget.__name__} vs {functionNdarrayOriginal.__name__}'
	)

def test_windowing_tensors_equivalence(device: str, lengthWindow: int) -> None:
	"""
	Verify all tensor-based windowing functions produce equivalent results to their numpy counterparts.
	"""
	prototype_tensorEquivalent(cosineWings, cosineWingsTensor, device, lengthWindow, ratioTaper=0.5)
	prototype_tensorEquivalent(equalPower, equalPowerTensor, device, lengthWindow, ratioTaper=0.3)
	prototype_tensorEquivalent(halfsine, halfsineTensor, device, lengthWindow)
	prototype_tensorEquivalent(tukey, tukeyTensor, device, lengthWindow, ratioTaper=0.7)

def test_tensor_special_cases(device: str) -> None:
	"""
	Verify special cases in tensor-based windowing functions.
	"""
	cosineWingsTensorResult = cosineWingsTensor(256, ratioTaper=0.0, device=torch.device(device))
	assert torch.allclose(cosineWingsTensorResult, torch.ones(256, device=torch.device(device), dtype=torch.float32)), (
		'cosineWingsTensor with ratioTaper=0.0 should produce all ones'
	)

	tukeyNormal = tukeyTensor(256, ratioTaper=0.5, device=torch.device(device))
	tukeyAlpha = tukeyTensor(256, alpha=0.5, device=torch.device(device))
	assert torch.allclose(tukeyNormal, tukeyAlpha), 'tukeyTensor should handle alpha parameter the same as ratioTaper'
