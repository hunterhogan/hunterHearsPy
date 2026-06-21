from __future__ import annotations

from hunterHearsPy import cosineWings, equalPower, halfsine, tukey
from tests.conftest import assert_allclose
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterHearsPy import WindowingFunction

@pytest.mark.parametrize('ratioTaper', tuple(map(pytest.param, (0.1, 0.05))))
@pytest.mark.parametrize('lengthSupport', tuple(map(pytest.param, (1024, 485100, 529200))))
def test_cosineWings(lengthSupport: int, ratioTaper: float, expected: WindowingFunction, rtol: float, atol: float) -> None:
	actual: WindowingFunction = cosineWings(lengthSupport=lengthSupport, ratioTaper=ratioTaper)

	assert_allclose(actual, expected, rtol, atol, 'cosineWings', lengthSupport, ratioTaper=ratioTaper)

@pytest.mark.parametrize('lengthSupport', [pytest.param(1024)])
@pytest.mark.parametrize('ratioTaper', [pytest.param(-0.1, id='ratioTaperBelowMinimum'), pytest.param(1.1, id='ratioTaperAboveMaximum')])
@pytest.mark.parametrize('expected', [pytest.param(ValueError, id='ValueError')])
def test_cosineWingsError(lengthSupport: int, ratioTaper: float, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		cosineWings(lengthSupport=lengthSupport, ratioTaper=ratioTaper)

@pytest.mark.parametrize('ratioTaper', tuple(map(pytest.param, (0.01, 0.002))))
@pytest.mark.parametrize('lengthSupport', tuple(map(pytest.param, (1024 * 4, 44100 * 18, 48000 * 15))))
def test_equalPower(lengthSupport: int, ratioTaper: float, expected: WindowingFunction, rtol: float, atol: float) -> None:
	actual: WindowingFunction = equalPower(lengthSupport=lengthSupport, ratioTaper=ratioTaper)

	assert_allclose(actual, expected, rtol, atol, 'equalPower', lengthSupport, ratioTaper=ratioTaper)

@pytest.mark.parametrize('lengthSupport', [pytest.param(2222)])
@pytest.mark.parametrize('ratioTaper', [pytest.param(-0.02, id='ratioTaperBelowMinimum'), pytest.param(5, id='ratioTaperAboveMaximum')])
@pytest.mark.parametrize('expected', [pytest.param(ValueError, id='ValueError')])
def test_equalPowerError(lengthSupport: int, ratioTaper: float, expected: type[Exception]) -> None:
	with pytest.raises(expected):
		equalPower(lengthSupport=lengthSupport, ratioTaper=ratioTaper)

@pytest.mark.parametrize('lengthSupport', tuple(map(pytest.param, (512, 1024, 2048))))
def test_halfsine(lengthSupport: int, expected: WindowingFunction, rtol: float, atol: float) -> None:
	actual: WindowingFunction = halfsine(lengthSupport=lengthSupport)

	assert_allclose(actual, expected, rtol, atol, 'halfsine', lengthSupport)

@pytest.mark.parametrize('lengthSupport', tuple(map(pytest.param, (512, 1024, 2048))))
@pytest.mark.parametrize('ratioTaper', tuple(map(pytest.param, (0.1, 0.05))))
@pytest.mark.parametrize('keywordArguments', [pytest.param({}, id='keywordArguments~None'), pytest.param({'alpha': 0.08}, id='keywordArguments~alpha0.08')])
def test_tukey(
	lengthSupport: int,
	ratioTaper: float,
	keywordArguments: dict[str, float],
	expected: WindowingFunction,
	rtol: float,
	atol: float,
) -> None:
	actual: WindowingFunction = tukey(lengthSupport=lengthSupport, ratioTaper=ratioTaper, **keywordArguments)

	assert_allclose(actual, expected, rtol, atol, 'tukey', lengthSupport, ratioTaper=ratioTaper, **keywordArguments)
