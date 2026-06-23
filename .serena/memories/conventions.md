# Conventions

- Do not duplicate HH Python, pytest, pandas, formatting, typing, naming, diagnostic, docstring, syntactic-clarity, or post-defensive rules in Serena. Use the corresponding global skill as the authority, then keep only project-specific facts here.
- Read `.editorconfig` and `pyproject.toml` for current formatting, tool, dependency, test, and coverage settings; Serena should name those authorities, not copy their values.
- Prefer existing package patterns over refactors. This codebase uses public flat imports from `hunterHearsPy` and project type aliases in `theTypes.py`.
- Do not introduce `__all__` in this project; the user does not use it. Keep flat namespace exports explicit in `hunterHearsPy/__init__.py`.
- Public imports should prefer root `hunterHearsPy` re-exports. Intra-package implementation may use relative imports when needed to prevent package-root circular imports, especially around `__init__.py` and optional modules.
- Keep `torch` optional: isolate tensor-specific runtime imports behind `contextlib.suppress` or `pytest.importorskip`; do not require torch for importing `hunterHearsPy` or non-tensor tests.
- Read `mem:instructions/python_conventions` for the skill map covering HH Python work.
- Read `mem:instructions/testing_conventions` before adding or reorganizing pytest tests; read `mem:tests/redesign_progress` before touching redesigned tests or expected data.
