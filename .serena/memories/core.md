# Core

- Python audio package using `src/` layout: importable package is `src/hunterHearsPy`.
- Public API is intentionally flat through `src/hunterHearsPy/__init__.py`; import user-facing functions/types from `hunterHearsPy` unless editing intra-package implementation.
- `torch` is optional. Root `hunterHearsPy` import must not fail only because `torch`/`torch.types` is missing; optional tensor root exports exist only when tensor imports succeed.
- Tests live in `tests/`; sample audio data lives under `tests/dataSamples/`; temporary test output is centralized under `tests/dataSamples/tmp` by `tests/conftest.py`.
- AST rewrite experiments/utilities live under `astTransformations/`; notebooks/notes live under `Z0Z_notes/`.
- Packaging, dependency, test, and coverage config live in `pyproject.toml`; basic formatting authority is `.editorconfig`. Read those files for current values.
- Read `mem:tech_stack` for package/dependency orientation, `mem:suggested_commands` for common commands, `mem:conventions` for project-specific conventions and skill routing, `mem:tests/redesign_progress` when continuing the in-flight pytest redesign, and `mem:task_completion` before closing coding tasks.
