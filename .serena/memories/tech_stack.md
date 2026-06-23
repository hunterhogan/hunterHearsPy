# Tech Stack

- Python package name is `hunterHearsPy`; source uses `src/` layout.
- `pyproject.toml` is authoritative for Python version, build backend, dependencies, optional dependency groups, pytest config, and coverage config. Read it instead of relying on Serena for current values.
- `torch` is optional in the runtime surface. Keep root `hunterHearsPy` import working without tensor dependencies.
- Source uses NumPy/SciPy-style audio and signal processing plus package filesystem helpers; inspect the current module before assuming a dependency or helper is still used.
