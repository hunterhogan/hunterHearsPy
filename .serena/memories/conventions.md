# Conventions

- Follow `.editorconfig`: Python/code files use UTF-8, LF, tabs, tab width 4, max line length 140, final newline, no trailing whitespace. Markdown/TOML/YAML use spaces, indent size 2, max line length 102.
- Prefer existing package patterns over refactors. This codebase uses camelCase identifiers, type-rich function signatures, NumPy-style type aliases in `theTypes.py`, and public flat imports from `hunterHearsPy`.
- Do not introduce `__all__` in this project; the user does not use it. Keep flat namespace exports explicit in `hunterHearsPy/__init__.py`.
- Public imports should prefer root `hunterHearsPy` re-exports. Intra-package implementation may use relative imports when needed to prevent package-root circular imports, especially around `__init__.py` and optional modules.
- Keep `torch` optional: isolate tensor-specific runtime imports behind `contextlib.suppress` or `pytest.importorskip`; do not require torch for importing `hunterHearsPy` or non-tensor tests.
- Do not add broad defensive guards inside already-validated internal code. Validate boundaries, then trust invariants and let invariant violations fail fast.
- Do not catch exceptions unless the catch has a concrete recovery, boundary translation, cleanup, or narrowing plan. Never catch only to announce/log/print that an exception happened.
- Do not add docstrings unless explicitly requested; when docstrings are requested, follow `mem:instructions/documentation_conventions`.
- Read `mem:instructions/python_conventions` for HH Python style, naming, formatting, typing, diagnostics, and control-flow rules.
- Read `mem:instructions/testing_conventions` before adding or reorganizing pytest tests.