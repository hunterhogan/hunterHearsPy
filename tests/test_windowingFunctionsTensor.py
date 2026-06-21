from __future__ import annotations

from hunterHearsPy import cosineWingsTensor, equalPowerTensor, halfsineTensor, tukeyTensor
from tests.conftest import assert_allclose, messageTestFailure
from typing import TYPE_CHECKING
import pytest
import torch

if TYPE_CHECKING:
	from hunterHearsPy import WindowingFunction
	from torch import Tensor
	from torch.types import Device

@pytest.mark.parametrize('ratioTaper', tuple(map(pytest.param, (1 / 44.1, 0.1))))
@pytest.mark.parametrize('lengthSupport', tuple(map(pytest.param, (44100 * 18, 48000 * 15))))
@pytest.mark.parametrize('dtype', tuple(map(pytest.param, (torch.float16, None))))
def test_cosineWingsTensor(lengthSupport: int, ratioTaper: float, device: Device | None, dtype: torch.dtype | None, expected: WindowingFunction, rtol: float, atol: float) -> None:
	actual: Tensor = cosineWingsTensor(lengthSupport=lengthSupport, ratioTaper=ratioTaper, device=device, dtype=dtype)
	expectedDevice: Device = torch.device(device=device or 'cpu')
	expectedDtype: torch.dtype = dtype or torch.float32
	function: str = 'cosineWingsTensor'

	assert actual.device == expectedDevice, messageTestFailure(actual.device, expectedDevice, function, lengthSupport, ratioTaper=ratioTaper, device=device, dtype=dtype)
	assert actual.dtype == expectedDtype, messageTestFailure(actual.dtype, expectedDtype, function, lengthSupport, ratioTaper=ratioTaper, device=device, dtype=dtype)
	assert_allclose(actual.cpu().numpy(), expected, rtol, atol, function, lengthSupport, ratioTaper=ratioTaper, device=device, dtype=dtype)

@pytest.mark.parametrize('lengthSupport', [pytest.param(31212012)])
@pytest.mark.parametrize('ratioTaper', [pytest.param(-2**32, id='ratioTaperBelowMinimum'), pytest.param(1.0000000001, id='ratioTaperAboveMaximum')])
@pytest.mark.parametrize('dtype', [pytest.param(None)])
@pytest.mark.parametrize('expected', [pytest.param(ValueError, id='ValueError')])
def test_cosineWingsTensorError(lengthSupport: int, ratioTaper: float, device: Device | None, dtype: torch.dtype | None, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		cosineWingsTensor(lengthSupport=lengthSupport, ratioTaper=ratioTaper, device=device, dtype=dtype)

@pytest.mark.parametrize('ratioTaper', tuple(map(pytest.param, (0.2, 0.0))))
@pytest.mark.parametrize('lengthSupport', tuple(map(pytest.param, (485100, 529200))))
@pytest.mark.parametrize('dtype', tuple(map(pytest.param, (None, torch.float32))))
def test_equalPowerTensor(lengthSupport: int, ratioTaper: float, device: Device | None, dtype: torch.dtype | None, expected: WindowingFunction, rtol: float, atol: float) -> None:
	actual: Tensor = equalPowerTensor(lengthSupport=lengthSupport, ratioTaper=ratioTaper, device=device, dtype=dtype)
	expectedDevice: Device = torch.device(device=device or 'cpu')
	expectedDtype: torch.dtype = dtype or torch.float32
	function: str = 'equalPowerTensor'

	assert actual.device == expectedDevice, messageTestFailure(actual.device, expectedDevice, function, lengthSupport, ratioTaper=ratioTaper, device=device, dtype=dtype)
	assert actual.dtype == expectedDtype, messageTestFailure(actual.dtype, expectedDtype, function, lengthSupport, ratioTaper=ratioTaper, device=device, dtype=dtype)
	assert_allclose(actual.cpu().numpy(), expected, rtol, atol, function, lengthSupport, ratioTaper=ratioTaper, device=device, dtype=dtype)

@pytest.mark.parametrize('lengthSupport', [pytest.param(31212012)])
@pytest.mark.parametrize('ratioTaper', [pytest.param(-0.1, id='ratioTaperBelowMinimum'), pytest.param(2**32, id='ratioTaperAboveMaximum')])
@pytest.mark.parametrize('dtype', [pytest.param(None)])
@pytest.mark.parametrize('expected', [pytest.param(ValueError, id='ValueError')])
def test_equalPowerTensorError(lengthSupport: int, ratioTaper: float, device: Device | None, dtype: torch.dtype | None, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		equalPowerTensor(lengthSupport=lengthSupport, ratioTaper=ratioTaper, device=device, dtype=dtype)

@pytest.mark.parametrize('lengthSupport', tuple(map(pytest.param, (147 * 2, 441 * 2, 2048))))
@pytest.mark.parametrize('dtype', tuple(map(pytest.param, (torch.float16, torch.float64))))
def test_halfsineTensor(lengthSupport: int, device: Device | None, dtype: torch.dtype | None, expected: WindowingFunction, rtol: float, atol: float) -> None:
	actual: Tensor = halfsineTensor(lengthSupport=lengthSupport, device=device, dtype=dtype)
	expectedDevice: Device = torch.device(device=device or 'cpu')
	expectedDtype: torch.dtype = dtype or torch.float32
	function: str = 'halfsineTensor'

	assert actual.device == expectedDevice, messageTestFailure(actual.device, expectedDevice, function, lengthSupport, device=device, dtype=dtype)
	assert actual.dtype == expectedDtype, messageTestFailure(actual.dtype, expectedDtype, function, lengthSupport, device=device, dtype=dtype)
	assert_allclose(actual.cpu().numpy(), expected, rtol, atol, function, lengthSupport, device=device, dtype=dtype)

@pytest.mark.parametrize('ratioTaper', tuple(map(pytest.param, (0.19, 1 / 64))))
@pytest.mark.parametrize('lengthSupport', tuple(map(pytest.param, (512, 1024, 4096))))
@pytest.mark.parametrize('keywordArguments', [pytest.param({}, id='keywordArguments~None'), pytest.param({'alpha': 0.08}, id='keywordArguments~alpha0.08')])
@pytest.mark.parametrize('dtype', tuple(map(pytest.param, (None, torch.float64))))
def test_tukeyTensor(lengthSupport: int, ratioTaper: float, device: Device | None, dtype: torch.dtype | None, keywordArguments: dict[str, float], expected: WindowingFunction, rtol: float, atol: float) -> None:
	actual: Tensor = tukeyTensor(lengthSupport=lengthSupport, ratioTaper=ratioTaper, device=device, dtype=dtype, **keywordArguments)
	expectedDevice: Device = torch.device(device=device or 'cpu')
	expectedDtype: torch.dtype = dtype or torch.float32
	function: str = 'tukeyTensor'

	assert actual.device == expectedDevice, messageTestFailure(actual.device, expectedDevice, function, lengthSupport, ratioTaper=ratioTaper, device=device, dtype=dtype, **keywordArguments)
	assert actual.dtype == expectedDtype, messageTestFailure(actual.dtype, expectedDtype, function, lengthSupport, ratioTaper=ratioTaper, device=device, dtype=dtype, **keywordArguments)
	assert_allclose(actual.cpu().numpy(), expected, rtol, atol, function, lengthSupport, ratioTaper=ratioTaper, device=device, dtype=dtype, **keywordArguments)
