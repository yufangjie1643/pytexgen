"""Adapters from pytexgen voxel data to the Voxel-ACDM solver.

The functions in this module avoid TexGen ``.inp/.eld/.ori`` round trips. They
convert :class:`VoxelGridData` from ``gpu_voxelizer`` into the structured phase
arrays expected by Voxel-ACDM, then call its batched homogenization solver.

The first supported production path is Voxel-ACDM's isotropic phase-LUT solver.
Anisotropic yarn support needs orientation fields from the voxelizer and should
be added on top of this adapter rather than by reintroducing text files.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

try:
    from .gpu_voxelizer import VoxelGridData, voxelize_textile_data
except ImportError:  # pragma: no cover - legacy TexGen package name
    from TexGen.gpu_voxelizer import VoxelGridData, voxelize_textile_data


@dataclass
class ACDMSolveResult:
    """Result returned by a direct Voxel-ACDM solve."""

    C_eff: np.ndarray
    engineering_constants: Dict[str, float]
    info: object
    timings: Dict[str, float]
    phase_ids: np.ndarray
    voxel_data: VoxelGridData
    solver: object


@dataclass
class ACDMFFTSolveResult:
    """Result returned by the Voxel-ACDM numpy FFT reference solver."""

    C_eff: np.ndarray
    engineering_constants: Dict[str, float]
    info: object
    timings: Dict[str, float]
    C_mandel_field: np.ndarray
    voxel_data: VoxelGridData
    solver: object


def _is_torch_tensor(value) -> bool:
    """Return true when ``value`` is a torch tensor-like object."""
    return hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "device")


def _to_numpy(value) -> np.ndarray:
    """Convert numpy-like or torch tensor data to numpy."""
    if _is_torch_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _candidate_acdm_roots(acdm_root: Optional[str] = None):
    """Yield plausible Voxel-ACDM roots in priority order."""
    if acdm_root:
        yield Path(acdm_root)
    env_root = os.environ.get("VOXEL_ACDM_ROOT")
    if env_root:
        yield Path(env_root)

    here = Path(__file__).resolve()
    for parent in here.parents:
        yield parent / "Voxel-ACDM"
        if parent.name.lower() == "pytexgen":
            yield parent.parent / "Voxel-ACDM"
    yield Path.cwd() / "Voxel-ACDM"
    yield Path.cwd().parent / "Voxel-ACDM"


def find_voxel_acdm_root(acdm_root: Optional[str] = None) -> Path:
    """Find a local Voxel-ACDM checkout.

    Parameters
    ----------
    acdm_root : str or None
        Explicit checkout path. When omitted, ``VOXEL_ACDM_ROOT`` and sibling
        directories near this package/current working directory are searched.
    """
    seen = set()
    for candidate in _candidate_acdm_roots(acdm_root):
        root = candidate.expanduser().resolve()
        if root in seen:
            continue
        seen.add(root)
        if (root / "femlib").is_dir() and (root / "README.md").is_file():
            return root
    raise FileNotFoundError(
        "Voxel-ACDM checkout not found. Pass acdm_root=... or set "
        "VOXEL_ACDM_ROOT to the repository root."
    )


def import_voxel_acdm(acdm_root: Optional[str] = None):
    """Add Voxel-ACDM to ``sys.path`` and import ``femlib``."""
    root = find_voxel_acdm_root(acdm_root)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return importlib.import_module("femlib")


def acdm_grid_shape(data: VoxelGridData) -> Tuple[int, int, int]:
    """Return Voxel-ACDM grid shape ``(Nz, Ny, Nx)`` for voxel data."""
    return data.shape


def acdm_voxel_size(data: VoxelGridData) -> Tuple[float, float, float]:
    """Return Voxel-ACDM voxel size ``(dx, dy, dz)`` as Python floats."""
    spacing = _to_numpy(data.voxel_size)
    return tuple(float(v) for v in spacing.tolist())


def acdm_domain_size(data: VoxelGridData) -> Tuple[float, float, float]:
    """Return Voxel-ACDM domain size ``(Lx, Ly, Lz)`` as Python floats."""
    aabb = _to_numpy(data.aabb)
    domain = aabb[1] - aabb[0]
    return tuple(float(v) for v in domain.tolist())


def _phase_id(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer phase ID in 0..15")
    result = int(value)
    if result < 0 or result > 15:
        raise ValueError("Voxel-ACDM isotropic phase ids must be in 0..15")
    return result


def _phase_overrides(
    yarn_phase_by_id: Optional[Mapping[int, int]],
) -> Dict[int, int]:
    if yarn_phase_by_id is None:
        return {}
    if not isinstance(yarn_phase_by_id, Mapping):
        raise ValueError("yarn_phase_by_id must be a mapping")
    result = {}
    for yarn_id, phase in yarn_phase_by_id.items():
        if (
            isinstance(yarn_id, bool)
            or not isinstance(yarn_id, Integral)
            or int(yarn_id) < 0
        ):
            raise ValueError("yarn phase mapping keys must be non-negative integers")
        result[int(yarn_id)] = _phase_id(
            f"phase for yarn {int(yarn_id)}",
            phase,
        )
    return result


def to_acdm_phase_ids(data: VoxelGridData,
                      *,
                      matrix_phase: int = 0,
                      yarn_phase: int = 1,
                      yarn_phase_by_id: Optional[Mapping[int, int]] = None,
                      batch: bool = True):
    """Convert pytexgen yarn ids to Voxel-ACDM int4 phase ids.

    Voxel-ACDM's isotropic phase-LUT path expects phase ids in ``0..15`` with
    shape ``(B,Nz,Ny,Nx)`` or ``(Nz,Ny,Nx)``. Matrix voxels are pytexgen
    ``yarn_id == -1`` and are mapped to ``matrix_phase``. Yarn voxels map to
    ``yarn_phase`` unless ``yarn_phase_by_id`` overrides individual yarn ids.
    """
    matrix_phase_value = _phase_id("matrix_phase", matrix_phase)
    yarn_phase_value = _phase_id("yarn_phase", yarn_phase)
    overrides = _phase_overrides(yarn_phase_by_id)
    grid = data.grid
    if _is_torch_tensor(grid):
        import torch

        phase = torch.full(
            grid.shape,
            matrix_phase_value,
            dtype=torch.uint8,
            device=grid.device,
        )
    else:
        phase = np.full(
            grid.shape,
            matrix_phase_value,
            dtype=np.uint8,
        )
    phase[grid >= 0] = yarn_phase_value
    for yarn_id, phase_id in sorted(overrides.items()):
        phase[grid == yarn_id] = phase_id
    return phase[None, ...] if batch else phase


def _any_backend(value) -> bool:
    if _is_torch_tensor(value):
        return bool(value.any().item())
    return bool(np.asarray(value).any())


def _used_phase_ids(
    data: VoxelGridData,
    *,
    matrix_phase: int,
    yarn_phase: int,
    yarn_phase_by_id: Optional[Mapping[int, int]],
):
    matrix_phase_value = _phase_id("matrix_phase", matrix_phase)
    yarn_phase_value = _phase_id("yarn_phase", yarn_phase)
    overrides = _phase_overrides(yarn_phase_by_id)
    grid = data.grid
    if _is_torch_tensor(grid):
        import torch

        overridden = torch.zeros_like(grid, dtype=torch.bool)
    else:
        overridden = np.zeros(grid.shape, dtype=bool)

    used = set()
    if _any_backend(grid < 0):
        used.add(matrix_phase_value)
    for yarn_id, phase_id in sorted(overrides.items()):
        mask = grid == yarn_id
        if _any_backend(mask):
            used.add(phase_id)
        overridden |= mask
    if _any_backend((grid >= 0) & ~overridden):
        used.add(yarn_phase_value)
    return used


def _torch_dtype(dtype: str):
    """Resolve a dtype string for torch."""
    import torch

    dtype = dtype.lower()
    if dtype in {"fp32", "float32"}:
        return torch.float32
    if dtype in {"fp64", "float64"}:
        return torch.float64
    raise ValueError('dtype must be one of "fp32", "float32", "fp64", or "float64"')


def _extract_eng_constants(femlib, C_eff: np.ndarray) -> Dict[str, float]:
    """Return engineering constants as plain floats."""
    eng = femlib.extract_engineering_constants(C_eff)
    return {key: float(value) for key, value in eng.items()}


def _validate_isotropic_material(
    name: str,
    material: Mapping[str, float],
) -> Dict[str, float]:
    """Validate an isotropic material mapping."""
    if (
        not isinstance(material, Mapping)
        or "E" not in material
        or "Nu" not in material
    ):
        raise ValueError(f"{name} must contain isotropic 'E' and 'Nu' entries")
    try:
        E_value = float(material["E"])
        nu_value = float(material["Nu"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} E and Nu must be finite numbers") from exc
    if not np.isfinite((E_value, nu_value)).all():
        raise ValueError(f"{name} E and Nu must be finite numbers")
    if E_value <= 0.0:
        raise ValueError(f"{name} E must be positive")
    if not -1.0 < nu_value < 0.5:
        raise ValueError(f"{name} Nu must satisfy -1 < Nu < 0.5")
    return {"E": E_value, "Nu": nu_value}


def _normalize_phase_materials(
    *,
    phase_materials: Optional[Mapping[int, Mapping[str, float]]],
    matrix_material: Optional[Mapping[str, float]],
    yarn_material: Optional[Mapping[str, float]],
    matrix_phase: int,
    yarn_phase: int,
) -> Dict[int, Dict[str, float]]:
    matrix_phase_value = _phase_id("matrix_phase", matrix_phase)
    yarn_phase_value = _phase_id("yarn_phase", yarn_phase)
    if phase_materials is not None:
        if matrix_material is not None or yarn_material is not None:
            raise ValueError(
                "phase_materials cannot be combined with legacy "
                "matrix_material or yarn_material"
            )
        if not isinstance(phase_materials, Mapping) or not phase_materials:
            raise ValueError("phase_materials must be a non-empty mapping")
        result = {}
        for phase, material in phase_materials.items():
            phase_value = _phase_id("phase_materials key", phase)
            result[phase_value] = _validate_isotropic_material(
                f"phase_materials[{phase_value}]",
                material,
            )
        return result

    if matrix_material is None or yarn_material is None:
        raise ValueError(
            "provide phase_materials or both matrix_material and yarn_material"
        )
    matrix_value = _validate_isotropic_material(
        "matrix_material",
        matrix_material,
    )
    yarn_value = _validate_isotropic_material(
        "yarn_material",
        yarn_material,
    )
    if matrix_phase_value == yarn_phase_value and matrix_value != yarn_value:
        raise ValueError(
            "matrix_phase and yarn_phase share one ID but materials differ"
        )
    return {
        matrix_phase_value: matrix_value,
        yarn_phase_value: yarn_value,
    }


def _build_phase_luts(
    phase_materials: Mapping[int, Mapping[str, float]],
    used_phase_ids,
):
    for phase in sorted(used_phase_ids):
        if phase not in phase_materials:
            raise ValueError(f"missing material for used phase {phase}")
    first_phase = min(phase_materials)
    first = phase_materials[first_phase]
    E_lut = np.full(16, float(first["E"]), dtype=np.float64)
    nu_lut = np.full(16, float(first["Nu"]), dtype=np.float64)
    for phase, material in sorted(phase_materials.items()):
        E_lut[phase] = float(material["E"])
        nu_lut[phase] = float(material["Nu"])
    return E_lut, nu_lut


def _make_fft_mesh_proxy(data: VoxelGridData):
    """Build the mesh-like object consumed by Voxel-ACDM's FFT utilities."""
    yarn_id = np.asarray(_to_numpy(data.grid), dtype=np.int32)
    shape = yarn_id.shape
    default_o1 = np.zeros(shape + (3,), dtype=np.float64)
    default_o2 = np.zeros(shape + (3,), dtype=np.float64)
    default_o1[..., 0] = 1.0
    default_o2[..., 1] = 1.0
    return SimpleNamespace(
        grid_shape=shape,
        yarn_id=yarn_id,
        orient1=default_o1,
        orient2=default_o2,
    )


def solve_acdm_fft_numpy_from_voxel_data(
    data: VoxelGridData,
    *,
    matrix_material: Mapping[str, float],
    yarn_material: Mapping[str, float],
    acdm_root: Optional[str] = None,
    method: str = "cg",
    scheme: str = "continuous",
    ref_strategy: str = "mean_C",
    tol: float = 1e-6,
    max_iter: int = 300,
    verbose: bool = False,
) -> ACDMFFTSolveResult:
    """Call Voxel-ACDM's pure numpy FFT reference solver.

    This path is useful on machines without CUDA/Triton. It accepts
    ``VoxelGridData`` directly and builds the Mandel stiffness field in memory,
    avoiding TexGen text-file export and re-read.

    The current pytexgen voxelizer does not yet expose yarn orientation fields,
    so this adapter intentionally uses the isotropic matrix/yarn path.
    """
    _validate_isotropic_material("matrix_material", matrix_material)
    _validate_isotropic_material("yarn_material", yarn_material)

    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    femlib = import_voxel_acdm(acdm_root)
    fft_mod = importlib.import_module("femlib.fft")
    timings["import_s"] = time.perf_counter() - t0

    mesh_proxy = _make_fft_mesh_proxy(data)

    t0 = time.perf_counter()
    C_mandel_field = fft_mod.build_stiffness_field(
        mesh_proxy,
        matrix_material,
        yarn_material,
        use_orientations=False,
        force_isotropic_yarn=True,
        verbose=verbose,
    )
    timings["build_stiffness_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    solver = fft_mod.FFTHomogenizer(
        C_mandel_field,
        acdm_domain_size(data),
        ref_strategy=ref_strategy,
        scheme=scheme,
    )
    timings["build_solver_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    C_eff, info = solver.compute_effective_stiffness(
        method=method,
        tol=tol,
        max_iter=max_iter,
    )
    timings["solve_s"] = time.perf_counter() - t0

    C_eff = np.asarray(C_eff, dtype=np.float64)
    return ACDMFFTSolveResult(
        C_eff=C_eff,
        engineering_constants=_extract_eng_constants(femlib, C_eff),
        info=info,
        timings=timings,
        C_mandel_field=C_mandel_field,
        voxel_data=data,
        solver=solver,
    )


def solve_acdm_isotropic_from_voxel_data(
    data: VoxelGridData,
    *,
    matrix_material: Optional[Mapping[str, float]] = None,
    yarn_material: Optional[Mapping[str, float]] = None,
    phase_materials: Optional[
        Mapping[int, Mapping[str, float]]
    ] = None,
    acdm_root: Optional[str] = None,
    device: str = "cuda",
    dtype: str = "fp32",
    precond: str = "fft",
    tol: float = 2e-6,
    max_iter: int = 2000,
    matrix_phase: int = 0,
    yarn_phase: int = 1,
    yarn_phase_by_id: Optional[Mapping[int, int]] = None,
    verbose: bool = False,
) -> ACDMSolveResult:
    """Call Voxel-ACDM's isotropic phase-LUT solver from ``VoxelGridData``.

    Parameters
    ----------
    data : VoxelGridData
        Structured voxel output from ``voxelize_textile_data``.
    matrix_material, yarn_material : mapping
        Isotropic material dictionaries with keys ``"E"`` and ``"Nu"``.
    acdm_root : str or None
        Local Voxel-ACDM repository root. If omitted, a sibling checkout is
        discovered automatically.
    device : str, default="cuda"
        Voxel-ACDM solver device. Its main FEM path requires CUDA.
    dtype : {"fp32", "fp64", "float32", "float64"}
        Solver precision.
    precond : {"fft", "fe-green", "jacobi", "none"}
        Preconditioner selection. ``fft`` and ``fe-green`` both call
        Voxel-ACDM's FE Green FFT preconditioner.
    """
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    phase_ids = to_acdm_phase_ids(
        data,
        matrix_phase=matrix_phase,
        yarn_phase=yarn_phase,
        yarn_phase_by_id=yarn_phase_by_id,
        batch=True,
    )
    timings["phase_ids_s"] = time.perf_counter() - t0
    normalized_materials = _normalize_phase_materials(
        phase_materials=phase_materials,
        matrix_material=matrix_material,
        yarn_material=yarn_material,
        matrix_phase=matrix_phase,
        yarn_phase=yarn_phase,
    )
    used_phases = _used_phase_ids(
        data,
        matrix_phase=matrix_phase,
        yarn_phase=yarn_phase,
        yarn_phase_by_id=yarn_phase_by_id,
    )
    E_lut, nu_lut = _build_phase_luts(
        normalized_materials,
        used_phases,
    )

    t0 = time.perf_counter()
    femlib = import_voxel_acdm(acdm_root)
    from femlib.fem_batched import FEMHomogenizerBatchedIsotropicPhases
    timings["import_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    solver = FEMHomogenizerBatchedIsotropicPhases.from_E_nu(
        phase_ids,
        E_lut,
        nu_lut,
        acdm_voxel_size(data),
        acdm_grid_shape(data),
        device=device,
        dtype=_torch_dtype(dtype),
        verbose=verbose,
    )
    timings["build_solver_s"] = time.perf_counter() - t0

    precond_norm = precond.lower()
    if precond_norm not in {"fft", "fe-green", "fe_green", "jacobi", "none"}:
        raise ValueError('precond must be "fft", "fe-green", "jacobi", or "none"')
    if precond_norm in {"fft", "fe-green", "fe_green"}:
        t0 = time.perf_counter()
        solver.enable_fft_precond(ref_strategy="mean_C", verbose=verbose)
        timings["precond_s"] = time.perf_counter() - t0
    else:
        timings["precond_s"] = 0.0

    t0 = time.perf_counter()
    C_eff_batch, info_batch = solver.compute_effective_stiffness(
        tol=tol, max_iter=max_iter
    )
    timings["solve_s"] = time.perf_counter() - t0

    C_eff = np.asarray(C_eff_batch[0], dtype=np.float64)
    return ACDMSolveResult(
        C_eff=C_eff,
        engineering_constants=_extract_eng_constants(femlib, C_eff),
        info=info_batch[0],
        timings=timings,
        phase_ids=phase_ids,
        voxel_data=data,
        solver=solver,
    )


def voxelize_and_solve_acdm_isotropic(
    textile,
    *,
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
    matrix_material: Mapping[str, float],
    yarn_material: Mapping[str, float],
    acdm_root: Optional[str] = None,
    voxel_backend: str = "numpy",
    voxel_device: Optional[str] = None,
    solver_device: str = "cuda",
    solver_dtype: str = "fp32",
    precond: str = "fft",
    tol: float = 2e-6,
    max_iter: int = 2000,
    chunk_voxels: int = 65536,
    workers: Optional[int] = None,
    aabb_pruning: bool = True,
    verbose: bool = False,
) -> ACDMSolveResult:
    """Voxelize a TexGen textile and directly solve it with Voxel-ACDM.

    This is the one-call isotropic path:

    ``CTextile -> VoxelGridData -> phase_ids -> FEMHomogenizerBatchedIsotropicPhases``.
    """
    data = voxelize_textile_data(
        textile,
        nx=nx, ny=ny, nz=nz,
        backend=voxel_backend,
        device=voxel_device,
        output="numpy",
        chunk_voxels=chunk_voxels,
        workers=workers,
        verbose=verbose,
        aabb_pruning=aabb_pruning,
    )
    return solve_acdm_isotropic_from_voxel_data(
        data,
        matrix_material=matrix_material,
        yarn_material=yarn_material,
        acdm_root=acdm_root,
        device=solver_device,
        dtype=solver_dtype,
        precond=precond,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
    )


def voxelize_and_solve_acdm_fft_numpy(
    textile,
    *,
    nx: int = 64,
    ny: int = 64,
    nz: int = 64,
    matrix_material: Mapping[str, float],
    yarn_material: Mapping[str, float],
    acdm_root: Optional[str] = None,
    method: str = "cg",
    scheme: str = "continuous",
    ref_strategy: str = "mean_C",
    tol: float = 1e-6,
    max_iter: int = 300,
    chunk_voxels: int = 65536,
    workers: Optional[int] = None,
    aabb_pruning: bool = True,
    verbose: bool = False,
) -> ACDMFFTSolveResult:
    """Voxelize a TexGen textile and solve with Voxel-ACDM's numpy FFT path."""
    data = voxelize_textile_data(
        textile,
        nx=nx, ny=ny, nz=nz,
        backend="numpy",
        output="numpy",
        chunk_voxels=chunk_voxels,
        workers=workers,
        verbose=verbose,
        aabb_pruning=aabb_pruning,
    )
    return solve_acdm_fft_numpy_from_voxel_data(
        data,
        matrix_material=matrix_material,
        yarn_material=yarn_material,
        acdm_root=acdm_root,
        method=method,
        scheme=scheme,
        ref_strategy=ref_strategy,
        tol=tol,
        max_iter=max_iter,
        verbose=verbose,
    )


__all__ = [
    "ACDMSolveResult",
    "ACDMFFTSolveResult",
    "find_voxel_acdm_root",
    "import_voxel_acdm",
    "acdm_grid_shape",
    "acdm_voxel_size",
    "acdm_domain_size",
    "to_acdm_phase_ids",
    "solve_acdm_fft_numpy_from_voxel_data",
    "solve_acdm_isotropic_from_voxel_data",
    "voxelize_and_solve_acdm_fft_numpy",
    "voxelize_and_solve_acdm_isotropic",
]
