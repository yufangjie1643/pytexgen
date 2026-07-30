# Repository Guidelines

## Project Structure & Module Organization

`Core/` contains the primary C++ textile geometry, meshing, and export code. Python-facing code is split between the legacy `TexGen/` modules, the installable package in `src/pytexgen/`, and SWIG bindings in `Python/Core.i`; generated `Python/Core.py` and `Python/Core_wrap.cxx` are intentionally committed. Third-party or supporting native libraries live in directories such as `Triangle/`, `tetgenlib/`, `CSparse/`, and `tinyxml/`. Put runnable examples and benchmarks in `script/`, root-level regression tests in `test_*.py`, legacy integration tests in `Python/Tests/`, and test fixtures in `Python/Tests/Data/`.

## Build, Test, and Development Commands

- `./build.sh` creates or reuses `.venv`, installs build dependencies with `uv`, compiles the extension, and verifies the import. Windows equivalents are `build.ps1` and `build.bat`.
- `pip install -e .` performs an editable scikit-build-core/CMake installation.
- `python -m build` creates wheel and source distributions under `dist/`.
- `python -m unittest discover -s . -p 'test_*.py'` runs the root unit-test suite after the extension is built.
- `python test_gpu_voxelizer_backends.py` runs the lightweight NumPy/optional Torch backend checks.

Keep generated meshes, benchmark results, and local environments out of commits; `.gitignore` already covers `build/`, `dist/`, `Saved_*/`, `*.inp`, `*.tg3`, and `.venv/`.

## Coding Style & Naming Conventions

Use four spaces in Python, `snake_case` for functions and variables, `PascalCase` for classes, and `test_*` for test methods. Follow the surrounding C++ style: paired `.h`/`.cpp` files, descriptive PascalCase type names, and the existing `C` prefix for public TexGen classes. No repository-wide formatter is configured, so keep diffs focused and match neighboring code. Modify `Python/Core.i` for binding changes; regenerate committed SWIG outputs only with `-DTEXGEN_REGENERATE_SWIG=ON`.

## Testing Guidelines

Tests use Python’s `unittest`; NumPy assertions are common for array results. Add focused regression coverage for fixes and exercise CPU paths by default. Gate Torch/CUDA behavior when the dependency or device is unavailable. Tests that need compiled bindings must run after installation. Avoid committing files produced by smoke tests.

## Commit & Pull Request Guidelines

Recent history favors short, imperative subjects, often Conventional Commit prefixes such as `feat:` and `chore:`. Use a focused subject (for example, `fix: preserve periodic node pairs`) and keep unrelated changes separate. Pull requests should explain motivation and behavior changes, list commands run, link relevant issues, and note platform or optional-backend coverage. Include screenshots only for visual output changes and never attach generated mesh data in the repository.
