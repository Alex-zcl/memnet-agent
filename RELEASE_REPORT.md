# Release report: memnet-agent 0.1.0

Date: 2026-08-04

## Artifacts

- Distribution name: `memnet-agent`
- Import package: `memnet_agent`
- Wheel: `dist/memnet_agent-0.1.0-py3-none-any.whl`
- Source distribution: `dist/memnet_agent-0.1.0.tar.gz`
- License: MIT
- Supported Python: 3.10+

## Verification completed

- 16 tests passed against the source tree.
- 16 tests passed after installing the built wheel from `dist/`.
- Wheel installed successfully into `site-packages` and completed an ask/learn/export/load smoke test.
- Source distribution built a wheel and completed an ask/learn smoke test.
- CLI entry point returned version 0.1.0.
- CLI imported and validated the original legacy SQLite graph.
- Wheel metadata, runtime dependencies, Python requirement and `py.typed` marker were inspected successfully.
- Both GitHub Actions YAML files were parsed successfully.
- Setuptools' distribution check completed during sdist creation.

The sandbox did not contain the standalone `twine` utility, so `twine check dist/*` was not executed here. It is included in the local release instructions and CI workflow.

## SHA-256

- Wheel: `4a475e23d08682b260d8c7b0dd397b59da14edb2c8ef4507ac82b20c8d18ec52`
- Source distribution: `06c5b7876c532357449f9f691e710acca0bfe1a1495a47684a3874a126c86e95`

## Publication status

The package has not been uploaded to PyPI. Publishing requires access to the owner's PyPI/GitHub profile. Follow `PUBLISHING.md` to create a pending Trusted Publisher and publish under the intended account.
