# HH Python Conventions

- Use `python-code-generation` as the broad HH Python implementation entrypoint.
- Use `python-identifiers` for naming, renaming, labels, filesystem semiotics, and serialized-key decisions.
- Use `python-formatting` for Python layout, whitespace, delimiter placement, quote style, and `.editorconfig`/`pyproject.toml` formatting authority.
- Use `python-type-annotations` for annotation-only work and type checker diagnostics.
- Use `python-diagnostic-messages` for exception, warning, logging, print, and assertion-message wording.
- Use `python-docstrings` only when docstrings or docstring quality are explicitly in scope.
- Use `python-syntactic-clarity` for expression readability, operator visibility, comparison orientation, semantic identifiers, and import clarity.
- Use `python-post-defensive` for post-validation invariant code, branch-cost review, and redundant guard removal.
- Use `pandas` for DataFrame, Series, Index, groupby, merge/join, reshape, aggregation, transformation, missing-data, and vectorization work.
- Project-specific additions remain in `mem:conventions`; this memory should not copy the skill rules.
