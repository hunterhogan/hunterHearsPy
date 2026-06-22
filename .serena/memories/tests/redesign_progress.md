# Test Redesign Progress

Status as of 2026-06-21 audit; verify worktree state before relying on these notes because the tests and sample data are actively being remade.

## Current Direction

- Migrate pytest tests toward `mem:instructions/testing_conventions` plus the bundled pytest skill: pytest-native parametrization, shared fixtures, deterministic samples under `tests/dataSamples/`, expected contracts under `tests/dataSamples/expected/`.
- Use this memory for project-specific mechanics only. The generic pytest skill remains authoritative for test shape, data gates, fixture discipline, and avoiding faux scenario machinery.

## Current Shared Test Modules

- `tests/_theSSOT.py` defines `pathDataSamples`, `pathDataSamplesExpected`, and `dtypeTokens`.
- `tests/__init__.py` re-exports `_theSSOT` names and assertion helpers from `tests/conftestAnnex.py`.
- `tests/conftest.py` currently contains fixtures, not assertion helpers:
  - tolerance fixtures: `approx_abs`, `approx_rel`, `atol`, `rtol`
  - parameter/data fixtures: `device`, `expected`, `pathFilename`, `waveform`
- `tests/conftestAnnex.py` contains assertion helpers:
  - `assertEqualTo`
  - `assert_approx`
  - `messageTestFailure`
  - `assert_array_equal`
  - `assert_allclose`
  - `messageTestFailure_ndarray`
- New tests should import helpers from `tests`, not directly from `tests.conftestAnnex` or `tests.conftest`.

## Assertion Helpers

- `assertEqualTo(actual, expected, function, *arguments, **keywordArguments)` uses plain `actual == expected` and formats failure with `messageTestFailure`. Use for scalar values and built-in/container equality where printing `repr(actual)` is useful.
- `assert_approx(actual, expected, pytest_rel, pytest_abs, function, *arguments, **keywordArguments)` uses `pytest.approx(expected, pytest_rel, pytest_abs, nan_ok=True)`. Use for scalar approximate contracts, not arrays.
- `messageTestFailure(...)` formats `function(arg0, key=value) = actual, but expected = ... .` Keyword arguments preserve insertion order. This is the scalar/general-object failure formatter.
- `assert_array_equal(actual, expected, function, *arguments, **keywordArguments)` uses `numpy.array_equal`. Use for exact array contracts such as expected audio samples loaded from `.npy`.
- `assert_allclose(actual, expected, rtol, atol, function, *arguments, **keywordArguments)` uses `numpy.allclose`. Use for approximate numeric array contracts.
- `messageTestFailure_ndarray(...)` only reports `shape` and `dtype`, not values. It assumes `actual` and `expected` have `.shape` and `.dtype`; using `assert_allclose` or `assert_array_equal` for scalar/list contracts can mask the intended assertion failure with an attribute error when the assertion fails.

## Mechanical Expected Fixture

- `expected` builds filenames as:
  - `request.function.__name__`
  - plus `__{parameterName}~{token}` for parameters that are both in the test function signature and in `request.node.callspec.params`
  - final suffix `.npy` under `tests/dataSamples/expected/`
- Tokens come from `hunterMakesPy.dataStructures.stringItUp` and are joined with no separator.
- Observed current tokens:
  - `None` -> `None`
  - `{}` -> `None`
  - `{'alpha': 0.08}` -> `alpha0.08`
  - `'cpu'` -> `cpu`
  - `torch.float16` -> `torch.float16`
  - `torch.float32` -> `torch.float32`
  - `torch.float64` -> `torch.float64`
- Indirect fixtures still use the original parametrized value in the expected filename. Example: `pathFilename` returns a `Path`, but expected filenames use the parametrized filename string.
- `expected` loads arrays with `numpy.load(..., mmap_mode='r', allow_pickle=False)`. Treat returned expected arrays as read-only.
- If a test function signature, parameter name, or parametrized value changes, regenerate or rename matching `.npy` files. Old expected files become unreachable.

## Current Active Redesigned Tests

- `tests/test_io.py` has active `test_readAudioFile`.
  - Parameters: `pathFilename`, `sampleRateDesired`, `dtype_str`, `expected`.
  - `pathFilename,dtype_str` are coupled explicit rows; `pathFilename` is indirect.
  - Current sample names use explicit dtype tokens such as `int16` and `float32`, not the older `s16`/`f32` shorthand.
  - Current expected file count for `test_readAudioFile__*.npy`: 14.
- `tests/test_windowingFunctions.py` has active NumPy windowing tests.
  - Expected file counts: cosine wings 6, equal power 6, halfsine 3, tukey 12.
- `tests/test_windowingFunctionsTensor.py` has active tensor-wrapper tests.
  - Expected file counts: cosine wings tensor 16, equal power tensor 16, halfsine tensor 12, tukey tensor 48; total tensor expected files 92.
  - There are currently no reachable `lengthWindow` tensor expected files; tensor tests now use `lengthSupport`.
- `tests/test_resample.py` is still a placeholder with an import and a comment block of candidate samples, no test functions.
- Old-style tests remain under `tests/test_old_*` and still contain fixtures, local helper patterns, and pre-redesign structures. Do not normalize them opportunistically while migrating an unrelated target.

## Tensor-Specific Rules

- `device` is a fixture-param dimension over `None` and `'cpu'`. Any test function that includes `device` runs both values.
- Because `device` participates in `expected` filename construction, expected arrays may be duplicated for `device~None` and `device~cpu` even when numeric values are identical.
- Tensor success tests should pass `dtype` through and assert `actual.dtype == (dtype or torch.float32)`.
- Normalize expected device with `torch.device(device=device or 'cpu')`; direct comparison of `actual.device` to `'cpu'` fails.
- For tensor expected values, match PyTorch conversion behavior. Creating NumPy `float16` arrays directly can differ from `torch.tensor(data=array, dtype=torch.float16).numpy()` enough to fail `numpy.allclose`.
- `ratioTaper=1.0000000001` already raises `ValueError` for cosine wings as of this audit.

## Guidance For Adding New Test Functions

1. Read the target implementation, current test file, `tests/conftest.py`, `tests/conftestAnnex.py`, `tests/_theSSOT.py`, and this memory before editing.
2. Mirror the target function parameters in the test signature, same names and order, with no defaults. Append `expected` after target parameters for success tests.
3. Use one success test and one `Error` test per target unless there is a strong reason not to.
4. Use stacked `@pytest.mark.parametrize` for independent dimensions. Use coupled explicit rows only when parameter values must travel together, such as `pathFilename,dtype_str`.
5. Before writing the test body, resolve the exact expected filename(s) that the `expected` fixture will request. Create or regenerate those `.npy` files first.
6. Use helpers from `tests`: `assert_array_equal` for exact arrays, `assert_allclose` for approximate arrays, `assertEqualTo` for exact scalar/object equality, and `assert_approx` for approximate scalar equality.
7. Pass every meaningful target input into the assertion message, using positional arguments for positional target inputs and keyword arguments for named target inputs.
8. Keep fixtures in `tests/conftest.py`. Put non-fixture helper functions in `tests/conftestAnnex.py` and re-export from `tests/__init__.py` when they are shared.
9. Do not hide target parameters in dictionaries unless the target itself accepts that dictionary, as `tukey(..., **keywordArguments)` currently does.
10. Run focused tests for the changed target first, then the active redesigned set when shared fixtures or expected loading changed.

## Validation Snapshot

- Focused active redesigned validation on 2026-06-21:
  - `pytest tests/test_io.py tests/test_windowingFunctions.py tests/test_windowingFunctionsTensor.py -n0`: 145 passed.
  - `pytest tests/test_io.py tests/test_windowingFunctions.py tests/test_windowingFunctionsTensor.py`: 145 passed with xdist.
- The local `.venv` Python launcher may point at an inaccessible Python 3.14 install. A workspace uv-managed CPython 3.14.6 with `PYTHONPATH=src;.;.venv/Lib/site-packages` was used for validation.
- SciPy compiled extensions previously failed to import inside the filesystem sandbox with `Access is denied`; focused pytest and expected-data generation may need unsandboxed execution in this environment.