#!/usr/bin/env python3
"""Runtime for the single-file PyTexGen source installer.

This file is rendered as ``__main__.py`` by ``build_portable_installer.py``.
Keep it compatible with the oldest supported interpreter, CPython 3.9.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence


BUNDLED_VERSION = "@PYTEXGEN_VERSION@"
BUNDLED_SDIST = "@PYTEXGEN_SDIST@"
BUNDLED_SHA256 = "@PYTEXGEN_SHA256@"

BUILD_REQUIREMENTS = (
    "pip>=23.1",
    "setuptools>=68",
    "wheel>=0.41",
    "cmake>=3.17",
    "ninja>=1.11",
    "scikit-build-core>=0.10",
    "numpy>=1.21",
)


class InstallerError(RuntimeError):
    """An actionable installer failure."""


def announce(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def run_checked(command: Sequence[str], description: str, cwd: Optional[Path] = None) -> None:
    try:
        subprocess.run(list(command), cwd=cwd, check=True)
    except FileNotFoundError as error:
        raise InstallerError(f"{description} failed: command not found: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        raise InstallerError(
            f"{description} failed with exit code {error.returncode}. "
            "Review the build output above for the first error."
        ) from error


def require_supported_python() -> None:
    if sys.version_info < (3, 9):
        raise InstallerError(
            f"PyTexGen {BUNDLED_VERSION} requires Python 3.9 or newer; "
            f"this interpreter is {platform.python_version()}."
        )
    if platform.python_implementation() != "CPython":
        raise InstallerError(
            "This installer currently supports CPython only because PyTexGen "
            "contains a CPython native extension."
        )


def compiler_hint() -> str:
    if sys.platform == "win32":
        return "Install Visual Studio 2022 Build Tools with the C++ workload."
    if sys.platform == "darwin":
        return "Run 'xcode-select --install' to install Apple's C++ toolchain."
    return "Install GCC/G++ or Clang with your operating system package manager."


def python_headers_hint() -> str:
    if sys.platform == "win32":
        return "Use the full CPython installer, which includes Python.h and import libraries."
    if sys.platform == "darwin":
        return "Use the python.org or Homebrew CPython distribution with development headers."
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        f"Install the Python development package (for example python{version}-dev "
        "on Debian/Ubuntu), or use a complete python.org/uv CPython distribution."
    )


def check_python_headers() -> None:
    include_dir = sysconfig.get_path("include")
    python_header = Path(include_dir) / "Python.h" if include_dir else None
    if python_header is None or not python_header.is_file():
        raise InstallerError(
            "Python development headers were not found for this interpreter. "
            + python_headers_hint()
        )


def check_native_toolchain() -> None:
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles(x86)")
        vswhere = None
        if program_files:
            candidate = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
            if candidate.is_file():
                vswhere = candidate
        if shutil.which("cl") is None and vswhere is None:
            print(
                "Warning: an MSVC compiler was not detected. " + compiler_hint(),
                file=sys.stderr,
                flush=True,
            )
        return

    candidates = ("c++", "g++", "clang++")
    if not any(shutil.which(name) for name in candidates):
        raise InstallerError("No C++ compiler was detected. " + compiler_hint())


def venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def prepare_python(current_environment: bool, requested_venv: Optional[Path]) -> tuple[Path, Optional[Path]]:
    if current_environment:
        return Path(sys.executable).resolve(), None

    venv_dir = (requested_venv or (Path.cwd() / ".pytexgen-venv")).expanduser().resolve()
    python_executable = venv_python(venv_dir)
    if venv_dir.exists() and not python_executable.is_file():
        raise InstallerError(
            f"The target exists but is not a usable virtual environment: {venv_dir}"
        )
    if not python_executable.is_file():
        announce(f"Creating virtual environment: {venv_dir}")
        run_checked(
            [sys.executable, "-m", "venv", str(venv_dir)],
            "virtual environment creation",
        )
    return python_executable, venv_dir


def ensure_pip(python_executable: Path) -> None:
    probe = subprocess.run(
        [str(python_executable), "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        announce("Bootstrapping pip")
        run_checked(
            [str(python_executable), "-m", "ensurepip", "--upgrade"],
            "pip bootstrap",
        )


def pip_index_options(index_url: Optional[str]) -> List[str]:
    if not index_url:
        return []
    return ["--index-url", index_url]


def install_build_requirements(python_executable: Path, index_url: Optional[str]) -> None:
    announce("Installing Python, CMake, Ninja, and NumPy build dependencies")
    command = [
        str(python_executable),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--disable-pip-version-check",
        *pip_index_options(index_url),
        *BUILD_REQUIREMENTS,
    ]
    run_checked(command, "build dependency installation")


def extract_payload(destination: Path) -> Path:
    archive_path = Path(sys.argv[0]).resolve()
    if not zipfile.is_zipfile(archive_path):
        raise InstallerError(f"The installer bundle is not a valid zipapp: {archive_path}")

    try:
        with zipfile.ZipFile(archive_path) as bundle:
            payload = bundle.read(BUNDLED_SDIST)
    except (KeyError, OSError, zipfile.BadZipFile) as error:
        raise InstallerError("The embedded PyTexGen source archive is missing or unreadable.") from error

    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != BUNDLED_SHA256:
        raise InstallerError(
            "The embedded source archive failed its SHA256 integrity check; "
            "download a fresh installer."
        )

    payload_path = destination / BUNDLED_SDIST
    payload_path.write_bytes(payload)
    return payload_path


def install_pytexgen(
    python_executable: Path,
    payload_path: Path,
    index_url: Optional[str],
) -> None:
    announce(f"Compiling and installing PyTexGen {BUNDLED_VERSION}")
    command = [
        str(python_executable),
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        "--no-build-isolation",
        "--disable-pip-version-check",
        *pip_index_options(index_url),
        str(payload_path),
    ]
    run_checked(command, "PyTexGen source build and installation", cwd=payload_path.parent)


def verify_installation(python_executable: Path, cwd: Path) -> None:
    announce("Verifying the installed geometry package")
    verification = (
        "import pytexgen; "
        "import pytexgen.batch; "
        "import pytexgen.gpu_voxelizer; "
        "import pytexgen.material_fields; "
        f"assert pytexgen.__version__ == {BUNDLED_VERSION!r}; "
        "print('PyTexGen', pytexgen.__version__, 'installed successfully')"
    )
    run_checked(
        [str(python_executable), "-c", verification],
        "installed package verification",
        cwd=cwd,
    )


def print_activation(venv_dir: Optional[Path], python_executable: Path) -> None:
    print("\nInstallation complete.")
    print(f"Python executable: {python_executable}")
    if venv_dir is None:
        print("PyTexGen was installed into the current Python environment.")
    elif sys.platform == "win32":
        print(f'PowerShell activation: & "{venv_dir / "Scripts" / "Activate.ps1"}"')
        print(f'Command Prompt activation: "{venv_dir / "Scripts" / "activate.bat"}"')
    else:
        activate = venv_dir / "bin" / "activate"
        print(f"Activate with: source {shlex.quote(str(activate))}")


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Compile and install the bundled PyTexGen {BUNDLED_VERSION} source package "
            "with pip."
        )
    )
    parser.add_argument(
        "--venv",
        type=Path,
        help="virtual environment path (default: ./.pytexgen-venv)",
    )
    parser.add_argument(
        "--current-environment",
        action="store_true",
        help="install into the Python environment running this installer",
    )
    parser.add_argument(
        "--index-url",
        help="optional pip package index URL used for dependency installation",
    )
    arguments = parser.parse_args(argv)
    if arguments.current_environment and arguments.venv is not None:
        parser.error("--venv and --current-environment cannot be used together")
    return arguments


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        if BUNDLED_VERSION.startswith("@"):
            raise InstallerError(
                "This is the installer source template, not a built .pyz bundle."
            )
        arguments = parse_arguments(argv)
        require_supported_python()
        print(
            f"PyTexGen {BUNDLED_VERSION} portable source installer\n"
            f"Host: {platform.system()} {platform.machine()}\n"
            f"Interpreter: CPython {platform.python_version()}"
        )
        check_native_toolchain()
        check_python_headers()
        python_executable, venv_dir = prepare_python(
            arguments.current_environment,
            arguments.venv,
        )
        ensure_pip(python_executable)
        install_build_requirements(python_executable, arguments.index_url)
        with tempfile.TemporaryDirectory(prefix="pytexgen-installer-") as temporary_dir:
            temporary_path = Path(temporary_dir)
            payload_path = extract_payload(temporary_path)
            install_pytexgen(python_executable, payload_path, arguments.index_url)
            verify_installation(python_executable, temporary_path)
        print_activation(venv_dir, python_executable)
        return 0
    except InstallerError as error:
        print(f"\nInstaller error: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
