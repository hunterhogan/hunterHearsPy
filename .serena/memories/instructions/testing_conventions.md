# HH Testing Conventions

- Pytest package-level settings belong in `pyproject.toml` where possible; fixtures/shared setup/temp files/shared data/hooks belong in `tests/conftest.py`.
- Never create fixtures outside `tests/conftest.py`. Fixture names use camelCase; temporary fixtures use `temp...`; mocks use `mock...`.
- One test function per function/class being tested. Combine related assertions that inspect the same computed result to avoid repeated setup and to show property relationships.
- Every test function should use `@pytest.mark.parametrize`; if a test does not use parametrize or a fixture, justify it. Single-case parametrize is acceptable to preserve expansion structure.
- Share expensive setup/computation via fixtures in `conftest.py`, not repeated local setup or DIY generator functions.
- Deterministic data only. Prefer static samples under `tests/dataSamples/`. That directory should contain data only: dicts, lists, tuples, constants, type aliases; no functions/classes/executable logic.
- Avoid random/faker data. Avoid special synthetic values with boundary/sentinel semantics (`0`, `1`, `-1`, empty strings/containers, alphabetical boundaries). Prefer distinctive non-contiguous values such as Fibonacci numbers, primes, card ranks, directions, country codes, Greek letters, or musical notes.
- Assertions must include descriptive messages with function/feature under test, actual and expected values, relevant inputs, and a final period. Use pytest-native assertions/messages rather than raising custom exceptions in tests.
- Centralize multi-parameter scenario configuration in one data structure; avoid scattered pickers and split config dicts.
- Test file naming: `test_<module_name>.py`; conftest remains `tests/conftest.py`; test function prefix `test_` is the pytest-required exception to camelCase.
- Optional dependencies in tests: use `pytest.importorskip` at module level for modules that exclusively test optional behavior; shared fixtures should skip only when used, not at conftest import time.
- Running tests: prefer MCP test runners when available; otherwise use `uv run pytest` from the repo root.