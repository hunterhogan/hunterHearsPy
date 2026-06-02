# Task Completion

- Before closing a coding task, verify no stale namespace references remain when relevant: `rg -n "Z0Z_tools|oldName"`.
- Run `uv run pytest` for behavioral verification. For narrow changes, also run the directly affected module, e.g. `uv run pytest tests/test_amplitude.py`.
- Run a root import smoke test after public API/package-init changes, checking representative exported names with `hasattr(hunterHearsPy, "symbolName")`; do not use `hunterHearsPy.__all__`.
- If packaging metadata changed, parse `pyproject.toml` with `tomllib`.
- If tests cannot be run, record the exact blocker and use available fallbacks only as fallbacks; do not imply diagnostics were a test run.
- For memory maintenance after onboarding or instruction changes, run or ask the user to run `serena memories check` from the repository root.