# Core

- Python audio package using `src/` layout: importable package is `src/hunterHearsPy`.
- Public API is intentionally flat through `src/hunterHearsPy/__init__.py`; import user-facing functions/types from `hunterHearsPy` unless editing intra-package implementation.
- Intra-package modules use relative imports to avoid package-root circular imports. Root package re-exports symbols from `theTypes`, `windowingFunctions`, `amplitude`, `autoRevert`, `clippingArrays`, `filesystemToolbox`, `ioAudio`, and optional tensor window functions.
- `torch` is optional. Root `hunterHearsPy` import must not fail only because `torch`/`torch.types` is missing; tensor names exist in `__all__` only when optional torch imports succeed.
- Tests live in `tests/`; sample audio data lives under `tests/dataSamples/`; temporary test output is centralized under `tests/dataSamples/tmp` by `tests/conftest.py`.
- AST rewrite experiments/utilities live under `astTransformations/`; notebooks/notes live under `Z0Z_notes/`.
- Packaging and test config live in `pyproject.toml`; basic formatting authority is `.editorconfig`.
- Read `mem:tech_stack` for packaging/dependency details, `mem:suggested_commands` for common commands, `mem:conventions` for project and HH coding rules, and `mem:task_completion` before closing coding tasks.