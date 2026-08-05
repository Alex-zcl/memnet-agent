# Changelog

## 0.1.2 — 2026-08-06

- Changed decay defaults from per-second to conservative daily decay.
- Lowered pruning and retained a minimum graph core.
- Protected explicit knowledge from pruning.
- Migrated untouched 0.1.x defaults automatically on load.
- Existing `storage_path` databases load automatically.
- Prevented merges across user, assistant and knowledge roles.
- Excluded assistant responses from retrieval by default.
- Removed quality-only unrelated retrieval matches.

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
