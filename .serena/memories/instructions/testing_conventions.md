# HH Testing Conventions

- Use the `pytest` skill as the authority for generic test shape, parametrization, static data, fixtures, data gates, expected values, warning/error partitions, and assertion contracts.
- Project package-level pytest settings live in `pyproject.toml`; read it for current values.
- Project shared fixtures, setup, temp-file policy, and hooks live in `tests/conftest.py`.
- Shared assertion helpers live in `tests/conftestAnnex.py` and are re-exported from `tests`.
- Static samples live under `tests/dataSamples/`; redesigned expected array contracts live under `tests/dataSamples/expected/`.
- Temporary fixtures use `temp...`; mock fixtures use `mock...`.
- `torch` is an optional user dependency, but not an optional test dependency. Never guard tests that require `torch` with `pytest.importorskip` or anything else that would hide the fact that torch is not installed.
- Read `mem:tests/redesign_progress` before changing redesigned tests, expected `.npy` files, or the expected fixture mechanics.
