# Suggested Commands

- Install/sync project env: `uv sync --all-groups`.
- Run all tests: `uv run pytest`.
- Run one test module: `uv run pytest tests/test_windowingFunctions.py`.
- Run with an existing venv on Windows: `.venv\Scripts\python.exe -m pytest`.
- Quick package import check: `uv run python -c "import hunterHearsPy; print(hunterHearsPy.__name__)"`.
- Check pyproject syntax without external tooling: `uv run python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('pyproject.toml parses')"`.
- After memory edits, user can run `serena memories check` from the project root.
