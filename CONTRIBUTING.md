# Contributing

1. Create a branch from `main`.
2. Install development dependencies with `python -m pip install -e ".[dev]"`.
3. Add or update tests for behavior changes.
4. Run `pytest`, `python -m build` and `python -m twine check dist/*`.
5. Open a focused pull request describing API and graph-format implications.

Public APIs follow semantic versioning. Changes to SQLite/JSON schemas must include backward-compatible loading or an explicit migration path.
