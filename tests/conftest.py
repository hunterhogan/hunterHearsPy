from __future__ import annotations

from hashlib import blake2b
from humpy_cytoolz.dicttoolz import keyfilter, merge
from hunterHearsPy import readAudioFile
from hunterMakesPy.dataStructures import stringItUp
from more_itertools import one
from soundfile import dtype_str as Options_dtype_str
from tests import pathDataSamples, pathDataSamplesExpected
from typing import cast, TYPE_CHECKING
import inspect
import numpy
import operator
import pytest

if TYPE_CHECKING:
	from collections.abc import Sequence
	from hunterHearsPy.theTypes import Waveform, 形ndarray
	from inspect import Parameter
	from pathlib import Path
	from pytest import FixtureRequest
	from torch.types import Device
	from types import MappingProxyType

# ================== Test-function parameters ======================================================

@pytest.fixture()
def approx_abs(request: FixtureRequest) -> float:
	"""The `abs` (***abs***olute tolerance) parameter value for `pytest.approx`."""
	return 1e-12

@pytest.fixture()
def approx_rel(request: FixtureRequest) -> float:
	"""The `rel` (***rel***ative tolerance) parameter value for `pytest.approx`."""
	return 1e-6

@pytest.fixture()
def atol(request: FixtureRequest) -> float:
	"""The `atol` (***a***bsolute ***tol***erance) parameter value for `numpy.allclose`."""
	return 1e-08

@pytest.fixture()
def expected(request: FixtureRequest) -> 形ndarray:
	"""Test-function and its parameters encoded in a `__` delimited filename.

	Each parameter created with `parametrize` (as opposed to, for example, `fixture`) is encoded as `{parameter}~{value}`.
	"""
	request_nodeParameters: MappingProxyType[str, Parameter] = cast('MappingProxyType[str, Parameter]', request.node.callspec.params)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
	filenameStem: str = '__'.join((
		str(request.function.__name__).removeprefix('test_')
		, *(f'{keyAndValue[0]}~{"".join(stringItUp(keyAndValue[1]) or ["None"])}'
			for keyAndValue in keyfilter(request_nodeParameters.keys().__contains__, merge(inspect.signature(request.function).parameters, request_nodeParameters)).items()
	)))
	if 251 < len(filenameStem):
		filenameStem = f'{str(request.function.__name__).removeprefix("test_")}__blake2b~{blake2b(filenameStem.encode(), digest_size=16).hexdigest()}'
	pathFilename: Path = pathDataSamplesExpected / f'{filenameStem}.npy'
	return numpy.load(pathFilename, mmap_mode='r', allow_pickle=False)

@pytest.fixture()
def rtol(request: FixtureRequest) -> float:
	"""The `rtol` (***r***elative ***tol***erance) parameter value for `numpy.allclose`."""
	return 1e-05

# ================== Parameter values to test against the package's `Callable` =====================

@pytest.fixture()
def CPUlimit(request: FixtureRequest) -> int:
	return 1

@pytest.fixture(params=tuple(map(pytest.param, (None, 'cpu'))))
def device(request: FixtureRequest) -> Device | None:
	return request.param

@pytest.fixture()
def dtype_str(pathFilename: Path) -> Options_dtype_str | None:
	return one(set(Options_dtype_str.__args__).intersection(pathFilename.stem.split('_')), too_short=None)

# TODO Resolving every actual pathFilename MUST be centralized in one function.
@pytest.fixture()
def listPathFilenames(request: FixtureRequest) -> tuple[Path, ...]:
	listFilenames: Sequence[str] = cast('Sequence[str]', request.node.callspec.params['listPathFilenames'])  # pyright: ignore[reportUnknownMemberType]
	return tuple(map(pathDataSamples.joinpath, listFilenames))

@pytest.fixture()
def pathFilename(request: FixtureRequest) -> Path:
	filename: str = request.param
	return pathDataSamples / filename

@pytest.fixture()
def sampleRateSource(pathFilename: Path) -> float:
	return float(one(filter(str.isdecimal, map(operator.itemgetter(slice(2, None)), filter(lambda string: string.startswith('Hz'), pathFilename.stem.split('_'))))))

@pytest.fixture()
def waveform(pathFilename: Path, sampleRateSource: float, dtype_str: Options_dtype_str | None) -> Waveform:
	return readAudioFile(pathFilename, sampleRateSource, dtype_str)
