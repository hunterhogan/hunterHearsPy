# Task Completion

- Run `uv run pytest` for behavioral verification when practical. For narrow changes, also run the directly affected test module.
- Run a root import smoke test after public API/package-init changes, checking representative exported names with `hasattr(hunterHearsPy, "symbolName")`; do not use `hunterHearsPy.__all__`.
- If packaging metadata changed, parse `pyproject.toml` with `tomllib`.
- If tests cannot be run, record the exact blocker and use available fallbacks only as fallbacks; do not imply diagnostics were a test run.
- For memory maintenance after onboarding or instruction changes, run or ask the user to run `serena memories check` from the repository root.
