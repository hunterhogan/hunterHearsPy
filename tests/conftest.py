from __future__ import annotations

from humpy_cytoolz.dicttoolz import keyfilter, merge
from hunterHearsPy import readAudioFile
from hunterMakesPy.dataStructures import stringItUp
from more_itertools import one
from soundfile import dtype_str as Options_dtype_str
from tests import dtypeTokens, pathDataSamples, pathDataSamplesExpected
from typing import TYPE_CHECKING
import inspect
import numpy
import operator
import pytest

if TYPE_CHECKING:
	from hunterHearsPy import Waveform, 形ndarray
	from pathlib import Path
	from torch.types import Device

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

@pytest.fixture()
def sampleRateSource(pathFilename: Path) -> float:
	tokens = pathFilename.stem.split('_')
	return float(one(filter(str.isdecimal, map(operator.itemgetter(slice(2, None)), filter(lambda string: string.startswith('Hz'), tokens)))))

@pytest.fixture()
def waveform(pathFilename: Path, sampleRateSource: float) -> Waveform:
	tokens = pathFilename.stem.split('_')
	dtype_str = one(filter(Options_dtype_str.__args__.__contains__, filter(dtypeTokens.__contains__, tokens)))
	return readAudioFile(pathFilename, sampleRateSource, dtype_str)  # pyright: ignore[reportArgumentType] # ty:ignore[invalid-argument-type]
