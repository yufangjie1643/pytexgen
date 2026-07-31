#!/usr/bin/env python3
"""Build a single-file cross-platform PyTexGen source installer."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import tempfile
import zipapp
from pathlib import Path
from typing import Optional, Sequence


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
RUNTIME_TEMPLATE = Path(__file__).with_name("portable_installer.py")


def project_version() -> str:
    pyproject = (REPOSITORY_DIR / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version = "([^"]+)"$', pyproject)
    if match is None:
        raise RuntimeError("Unable to read the project version from pyproject.toml")
    return match.group(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    version = project_version()
    release_dir = REPOSITORY_DIR / "dist" / f"release-{version}"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sdist",
        type=Path,
        default=release_dir / f"pytexgen-{version}.tar.gz",
        help="source distribution to embed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=release_dir / f"pytexgen-{version}-installer.pyz",
        help="output zipapp path",
    )
    return parser.parse_args(argv)


def render_runtime(version: str, payload_name: str, payload_hash: str) -> str:
    runtime = RUNTIME_TEMPLATE.read_text(encoding="utf-8")
    replacements = {
        "@PYTEXGEN_VERSION@": version,
        "@PYTEXGEN_SDIST@": payload_name,
        "@PYTEXGEN_SHA256@": payload_hash,
    }
    for marker, value in replacements.items():
        runtime = runtime.replace(marker, value)
    if "@PYTEXGEN_" in runtime:
        raise RuntimeError("An installer template marker was not replaced")
    return runtime


def build_installer(sdist: Path, output: Path) -> str:
    version = project_version()
    sdist = sdist.expanduser().resolve()
    output = output.expanduser().resolve()
    if not sdist.is_file():
        raise FileNotFoundError(f"Source distribution not found: {sdist}")
    if not RUNTIME_TEMPLATE.is_file():
        raise FileNotFoundError(f"Installer runtime template not found: {RUNTIME_TEMPLATE}")

    payload_name = f"pytexgen-{version}.tar.gz"
    payload_hash = sha256_file(sdist)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_dir = REPOSITORY_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="portable-installer-", dir=build_dir) as temporary_dir:
        temporary_path = Path(temporary_dir)
        staging = temporary_path / "app"
        staging.mkdir()
        (staging / "__main__.py").write_text(
            render_runtime(version, payload_name, payload_hash),
            encoding="utf-8",
        )
        shutil.copy2(sdist, staging / payload_name)
        (staging / "INSTALLER-README.txt").write_text(
            (
                f"PyTexGen {version} portable source installer\n\n"
                f"Run: python {output.name}\n"
                "Optional: --venv PATH, --current-environment, --index-url URL\n\n"
                "Python dependencies, CMake, and Ninja are installed with pip.\n"
                "A system C++ compiler is still required: MSVC Build Tools on Windows,\n"
                "Xcode Command Line Tools on macOS, or GCC/Clang on Linux.\n"
            ),
            encoding="utf-8",
        )

        temporary_bundle = temporary_path / output.name
        zipapp.create_archive(
            staging,
            target=temporary_bundle,
            interpreter="/usr/bin/env python3",
            compressed=True,
        )
        temporary_bundle.replace(output)

    if os.name != "nt":
        output.chmod(output.stat().st_mode | 0o111)
    installer_hash = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{installer_hash}  {output.name}\n",
        encoding="utf-8",
    )
    return installer_hash


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(argv)
    installer_hash = build_installer(arguments.sdist, arguments.output)
    print(f"Portable installer: {arguments.output.expanduser().resolve()}")
    print(f"SHA256: {installer_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
