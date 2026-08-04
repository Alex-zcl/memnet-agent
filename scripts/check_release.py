from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*command: str) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version_file = (ROOT / "src/memnet_agent/version.py").read_text(encoding="utf-8")
    project_version = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    module_version = re.search(r'__version__ = "([^"]+)"', version_file)
    if not project_version or not module_version or project_version.group(1) != module_version.group(1):
        raise SystemExit("Version mismatch between pyproject.toml and version.py")
    shutil.rmtree(ROOT / "dist", ignore_errors=True)
    run(sys.executable, "-m", "pytest")
    run(sys.executable, "-m", "build")
    distributions = [str(path) for path in sorted((ROOT / "dist").glob("*"))]
    run(sys.executable, "-m", "twine", "check", *distributions)


if __name__ == "__main__":
    main()
