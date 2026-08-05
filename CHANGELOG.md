# Changelog

## 0.1.1 — 2026-08-05

- Fixed default `UTC` sleep configuration on Windows systems without a system IANA timezone database.
- Added `tzdata` as an automatic Windows-only dependency for named timezones such as `Europe/Moscow`.
- Added clearer validation errors and regression tests for missing timezone data.

## 0.1.0 — 2026-08-04

- Public `MemoryAgent(model=...)` API.
- Callable and object-method model adapters.
- Associative free-text retrieval and automatic graph updates.
- Silent ingestion with `learn()` / `ingest()`.
- Idle, scheduled and worker sleep modes.
- Iterative JSONL training dataset generation.
- SQLite, JSON, GraphML and zip-bundle export.
- External SQLite, JSON and bundle loading.
- CLI for validation, conversion, statistics and dataset generation.
