"""Voxelization entry points for modern textile models."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .compat import load_gpu_voxelizer


def voxelize_model_data(
    model,
    resolution=(64, 64, 64),
    *,
    backend: str = "numpy",
    device=None,
    dtype: str = "float32",
    workers: int | str | None = "auto",
    fast_path: bool = True,
    **kwargs,
):
    """Voxelize a modern textile model and return ``VoxelGridData``.

    ``workers="auto"`` uses a conservative numpy policy: serial for tiny grids,
    two workers around 64^3, and at most four workers from 128^3 upward. Wider
    pools are still accepted explicitly for local benchmarks.
    """
    backend = backend.lower()
    if backend == "triton":
        raise NotImplementedError(
            "backend='triton' is reserved; use backend='torch' until a Triton "
            "kernel is implemented"
        )
    if backend not in {"numpy", "torch", "auto"}:
        raise ValueError('backend must be one of "numpy", "torch", "auto", or "triton"')
    nx, ny, nz = (int(value) for value in resolution)
    output = kwargs.pop("output", "backend" if backend == "torch" else "numpy")
    output = output.lower()
    workers = _resolve_modern_workers(backend, workers, resolution=(nx, ny, nz))
    if fast_path and backend in {"numpy", "torch"} and _is_plain_weave_model(model):
        fast_kwargs = {"chunk_voxels", "include_centers", "aabb_pruning", "progress"}
        if set(kwargs).issubset(fast_kwargs):
            return _voxelize_plain_weave_fast(
                model,
                resolution=(nx, ny, nz),
                backend=backend,
                device=device,
                dtype=dtype,
                workers=workers,
                output=output,
                include_centers=bool(kwargs.get("include_centers", False)),
                aabb_pruning=bool(kwargs.get("aabb_pruning", True)),
            )

    textile = model.to_model() if hasattr(model, "to_model") else model
    gv = load_gpu_voxelizer()
    snapshots = textile.to_snapshots(gv)
    return gv.voxelize_snapshots_data(
        snapshots,
        textile.aabb,
        nx=nx,
        ny=ny,
        nz=nz,
        backend=backend,
        device=device,
        dtype=dtype,
        output=output,
        verbose=False,
        workers=workers,
        **kwargs,
    )


def _resolve_modern_workers(
    backend: str,
    workers: int | str | None,
    resolution: tuple[int, int, int],
) -> int | None:
    """Resolve modern API worker policy before calling the legacy voxelizer."""
    if workers is None:
        return None
    if isinstance(workers, str):
        if workers.lower() != "auto":
            raise ValueError('workers must be an integer, None, or "auto"')
        return _auto_numpy_workers(resolution) if backend in {"numpy", "auto"} else None
    if workers < 1:
        raise ValueError("workers must be >= 1")
    return workers


def _auto_numpy_workers(resolution: tuple[int, int, int]) -> int:
    voxel_count = resolution[0] * resolution[1] * resolution[2]
    if voxel_count < 64 ** 3:
        return 1
    if voxel_count < 128 ** 3:
        return max(1, min(os.cpu_count() or 1, 2))
    return max(1, min(os.cpu_count() or 1, 4))


def _is_plain_weave_model(model) -> bool:
    from .weave import PlainWeave2D

    return isinstance(model, PlainWeave2D)


def _voxelize_plain_weave_fast(
    model,
    *,
    resolution: tuple[int, int, int],
    backend: str,
    device,
    dtype: str,
    workers: int | None,
    output: str,
    include_centers: bool,
    aabb_pruning: bool,
):
    gv = load_gpu_voxelizer()
    np_dtype = _numpy_dtype(dtype)
    aabb = _plain_weave_aabb(model)
    t0 = time.perf_counter()
    if backend == "torch":
        data = _voxelize_plain_weave_torch(
            gv,
            model,
            resolution=resolution,
            device=device,
            dtype=dtype,
            include_centers=include_centers,
            aabb_pruning=aabb_pruning,
        )
    else:
        worker_count = 1 if workers is None else int(workers)
        yarn_id = _classify_plain_weave_numpy(
            model,
            resolution=resolution,
            dtype=np_dtype,
            workers=worker_count,
        )
        centers = _structured_centers_numpy(aabb, resolution, np_dtype) if include_centers else None
        data = gv.VoxelGridData(
            yarn_id=yarn_id,
            aabb=aabb,
            resolution=resolution,
            backend="numpy",
            device="cpu",
            workers=max(1, worker_count),
            dtype=dtype,
            timings={"extract": 0.0, "pack": 0.0, "classify": 0.0},
            centers=centers,
            aabb_pruning=aabb_pruning,
            storage="numpy",
        )
    data.timings["classify"] = time.perf_counter() - t0
    return _coerce_output(data, output, device=device)


def _numpy_dtype(dtype: str):
    try:
        return {"float32": np.float32, "float64": np.float64}[dtype]
    except KeyError as exc:
        raise ValueError('dtype must be "float32" or "float64"') from exc


def _plain_weave_aabb(model) -> np.ndarray:
    z_margin = 0.1 * model.yarn_height
    return np.array(
        [
            [-0.5 * model.spacing, -0.5 * model.spacing, -z_margin],
            [
                (model.width - 0.5) * model.spacing,
                (model.height - 0.5) * model.spacing,
                model.thickness + z_margin,
            ],
        ],
        dtype=np.float64,
    )


def _voxel_axes_numpy(
    aabb: np.ndarray,
    resolution: tuple[int, int, int],
    dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = resolution
    lo, hi = aabb
    xs = (lo[0] + (np.arange(nx, dtype=np.float64) + 0.5) * (hi[0] - lo[0]) / nx).astype(dtype)
    ys = (lo[1] + (np.arange(ny, dtype=np.float64) + 0.5) * (hi[1] - lo[1]) / ny).astype(dtype)
    zs = (lo[2] + (np.arange(nz, dtype=np.float64) + 0.5) * (hi[2] - lo[2]) / nz).astype(dtype)
    return xs, ys, zs


def _structured_centers_numpy(aabb: np.ndarray, resolution: tuple[int, int, int], dtype):
    xs, ys, zs = _voxel_axes_numpy(aabb, resolution, dtype)
    gz, gy, gx = np.meshgrid(zs, ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)


def _classify_plain_weave_numpy(
    model,
    *,
    resolution: tuple[int, int, int],
    dtype,
    workers: int,
) -> np.ndarray:
    aabb = _plain_weave_aabb(model)
    xs, ys, zs = _voxel_axes_numpy(aabb, resolution, dtype)
    worker_count = max(1, min(int(workers), resolution[2]))
    if worker_count == 1:
        return _classify_plain_weave_numpy_zrange(model, xs, ys, zs, dtype).ravel()

    ranges = _split_ranges(resolution[2], worker_count)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        slices = list(
            executor.map(
                lambda bounds: _classify_plain_weave_numpy_zrange(
                    model, xs, ys, zs[bounds[0]:bounds[1]], dtype
                ),
                ranges,
            )
        )
    return np.concatenate(slices, axis=0).ravel()


def _split_ranges(length: int, parts: int) -> list[tuple[int, int]]:
    parts = max(1, min(parts, length))
    base, extra = divmod(length, parts)
    ranges = []
    start = 0
    for index in range(parts):
        stop = start + base + (1 if index < extra else 0)
        ranges.append((start, stop))
        start = stop
    return ranges


def _classify_plain_weave_numpy_zrange(model, xs, ys, zs, dtype) -> np.ndarray:
    nz, ny, nx = len(zs), len(ys), len(xs)
    best_dist = np.full((nz, ny, nx), np.inf, dtype=dtype)
    yarn_id = np.full((nz, ny, nx), -1, dtype=np.int32)
    polygon = _ellipse_polygon(model.yarn_width, model.yarn_height, dtype=dtype)
    half_width = dtype(model.yarn_width * 0.5)
    half_height = dtype(model.yarn_height * 0.5)

    x_nodes = np.linspace(0.0, model.width * model.spacing, model.width + 1, dtype=dtype)
    y_nodes = np.linspace(0.0, model.height * model.spacing, model.height + 1, dtype=dtype)

    for row in range(model.height):
        y_center = dtype(row * model.spacing)
        z_nodes = np.asarray(
            [model._cell_z(column % model.width, row, "x") for column in range(model.width + 1)],
            dtype=dtype,
        )
        nn, dx_nn, dz_nn = _nearest_node_components(xs, zs, x_nodes, z_nodes)
        _update_numpy_yarn(
            yarn_id,
            best_dist,
            yarn_index=row,
            u_axis=-(ys - y_center),
            v_grid=dz_nn,
            t_grid=dx_nn,
            nearest_grid=nn,
            polygon=polygon,
            half_width=half_width,
            half_height=half_height,
            mode="x",
        )

    for column in range(model.width):
        x_center = dtype(column * model.spacing)
        z_nodes = np.asarray(
            [model._cell_z(column, row % model.height, "y") for row in range(model.height + 1)],
            dtype=dtype,
        )
        nn, dy_nn, dz_nn = _nearest_node_components(ys, zs, y_nodes, z_nodes)
        _update_numpy_yarn(
            yarn_id,
            best_dist,
            yarn_index=model.height + column,
            u_axis=xs - x_center,
            v_grid=dz_nn,
            t_grid=dy_nn,
            nearest_grid=nn,
            polygon=polygon,
            half_width=half_width,
            half_height=half_height,
            mode="y",
        )

    return yarn_id


def _nearest_node_components(axis_values, z_values, axis_nodes, z_nodes):
    axis_delta = axis_values[:, None] - axis_nodes[None, :]
    z_delta = z_values[:, None] - z_nodes[None, :]
    d2 = z_delta[:, None, :] ** 2 + axis_delta[None, :, :] ** 2
    nn = np.argmin(d2, axis=2)
    axis_nn = axis_values[None, :] - axis_nodes[nn]
    z_nn = z_values[:, None] - z_nodes[nn]
    nearest_d2 = np.take_along_axis(d2, nn[..., None], axis=2).squeeze(axis=2)
    return nearest_d2, axis_nn, z_nn


def _update_numpy_yarn(
    yarn_id,
    best_dist,
    *,
    yarn_index: int,
    u_axis,
    v_grid,
    t_grid,
    nearest_grid,
    polygon,
    half_width,
    half_height,
    mode: str,
) -> None:
    if mode == "x":
        bbox = (np.abs(u_axis)[None, :, None] <= half_width) & (
            np.abs(v_grid)[:, None, :] <= half_height
        )
        iz, iy, ix = np.nonzero(bbox)
        if iz.size == 0:
            return
        u = u_axis[iy]
        v = v_grid[iz, ix]
        t = t_grid[iz, ix]
        d2 = nearest_grid[iz, ix] + u ** 2
    else:
        bbox = (np.abs(u_axis)[None, None, :] <= half_width) & (
            np.abs(v_grid)[:, :, None] <= half_height
        )
        iz, iy, ix = np.nonzero(bbox)
        if iz.size == 0:
            return
        u = u_axis[ix]
        v = v_grid[iz, iy]
        t = t_grid[iz, iy]
        d2 = nearest_grid[iz, iy] + u ** 2

    inside = _points_in_polygon_numpy(u, v, polygon)
    if not np.any(inside):
        return
    iz, iy, ix = iz[inside], iy[inside], ix[inside]
    t = t[inside]
    dist = np.sqrt(d2[inside]) + np.abs(t) * 0.1
    update = dist < best_dist[iz, iy, ix]
    if np.any(update):
        iz, iy, ix = iz[update], iy[update], ix[update]
        best_dist[iz, iy, ix] = dist[update]
        yarn_id[iz, iy, ix] = int(yarn_index)


def _ellipse_polygon(width: float, height: float, *, dtype) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False, dtype=np.float64)
    polygon = np.column_stack(
        [0.5 * float(width) * np.cos(angles), 0.5 * float(height) * np.sin(angles)]
    )
    return np.vstack([polygon, polygon[:1]]).astype(dtype, copy=False)


def _points_in_polygon_numpy(u: np.ndarray, v: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    poly = polygon[:-1]
    p_next = polygon[1:]
    u2 = u[:, None]
    v2 = v[:, None]
    x1 = poly[:, 0]
    y1 = poly[:, 1]
    x2 = p_next[:, 0]
    y2 = p_next[:, 1]
    cond1 = (y1 > v2) != (y2 > v2)
    denom = y2 - y1
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    xi = x1 + (v2 - y1) * (x2 - x1) / denom
    hits = (cond1 & (u2 < xi)).sum(axis=-1)
    return (hits % 2) == 1


def _voxelize_plain_weave_torch(
    gv,
    model,
    *,
    resolution: tuple[int, int, int],
    device,
    dtype: str,
    include_centers: bool,
    aabb_pruning: bool,
):
    torch = gv._require_torch()
    torch_dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    aabb_np = _plain_weave_aabb(model)
    yarn_grid = _classify_plain_weave_torch_grid(
        model,
        resolution=resolution,
        device=device,
        torch_dtype=torch_dtype,
        torch=torch,
    )
    centers = None
    if include_centers:
        centers = torch.as_tensor(
            _structured_centers_numpy(aabb_np, resolution, _numpy_dtype(dtype)),
            device=device,
            dtype=torch_dtype,
        )
    return gv.VoxelGridData(
        yarn_id=yarn_grid.reshape(-1),
        aabb=torch.as_tensor(aabb_np, device=device, dtype=torch_dtype),
        resolution=resolution,
        backend="torch",
        device=str(device),
        workers=1,
        dtype=dtype,
        timings={"extract": 0.0, "pack": 0.0, "classify": 0.0},
        centers=centers,
        aabb_pruning=aabb_pruning,
        storage="torch",
    )


def _classify_plain_weave_torch_grid(model, *, resolution, device, torch_dtype, torch):
    nx, ny, nz = resolution
    aabb = _plain_weave_aabb(model)
    xs_np, ys_np, zs_np = _voxel_axes_numpy(aabb, resolution, np.float64)
    xs = torch.as_tensor(xs_np, device=device, dtype=torch_dtype)
    ys = torch.as_tensor(ys_np, device=device, dtype=torch_dtype)
    zs = torch.as_tensor(zs_np, device=device, dtype=torch_dtype)
    best_dist = torch.full((nz, ny, nx), float("inf"), device=device, dtype=torch_dtype)
    yarn_id = torch.full((nz, ny, nx), -1, device=device, dtype=torch.int32)
    polygon = torch.as_tensor(
        _ellipse_polygon(model.yarn_width, model.yarn_height, dtype=np.float64),
        device=device,
        dtype=torch_dtype,
    )
    half_width = float(model.yarn_width * 0.5)
    half_height = float(model.yarn_height * 0.5)
    x_nodes = torch.linspace(
        0.0, model.width * model.spacing, model.width + 1, device=device, dtype=torch_dtype
    )
    y_nodes = torch.linspace(
        0.0, model.height * model.spacing, model.height + 1, device=device, dtype=torch_dtype
    )

    for row in range(model.height):
        z_nodes = torch.as_tensor(
            [model._cell_z(column % model.width, row, "x") for column in range(model.width + 1)],
            device=device,
            dtype=torch_dtype,
        )
        nn, dx_nn, dz_nn = _nearest_node_components_torch(xs, zs, x_nodes, z_nodes, torch)
        _update_torch_yarn(
            yarn_id,
            best_dist,
            yarn_index=row,
            u_axis=-(ys - row * model.spacing),
            v_grid=dz_nn,
            t_grid=dx_nn,
            nearest_grid=nn,
            polygon=polygon,
            half_width=half_width,
            half_height=half_height,
            mode="x",
            torch=torch,
        )

    for column in range(model.width):
        z_nodes = torch.as_tensor(
            [model._cell_z(column, row % model.height, "y") for row in range(model.height + 1)],
            device=device,
            dtype=torch_dtype,
        )
        nn, dy_nn, dz_nn = _nearest_node_components_torch(ys, zs, y_nodes, z_nodes, torch)
        _update_torch_yarn(
            yarn_id,
            best_dist,
            yarn_index=model.height + column,
            u_axis=xs - column * model.spacing,
            v_grid=dz_nn,
            t_grid=dy_nn,
            nearest_grid=nn,
            polygon=polygon,
            half_width=half_width,
            half_height=half_height,
            mode="y",
            torch=torch,
        )
    return yarn_id


def _nearest_node_components_torch(axis_values, z_values, axis_nodes, z_nodes, torch):
    axis_delta = axis_values[:, None] - axis_nodes[None, :]
    z_delta = z_values[:, None] - z_nodes[None, :]
    d2 = z_delta[:, None, :] ** 2 + axis_delta[None, :, :] ** 2
    nn = torch.argmin(d2, dim=2)
    axis_nn = axis_values[None, :] - axis_nodes[nn]
    z_nn = z_values[:, None] - z_nodes[nn]
    return d2.gather(2, nn.unsqueeze(-1)).squeeze(-1), axis_nn, z_nn


def _update_torch_yarn(
    yarn_id,
    best_dist,
    *,
    yarn_index: int,
    u_axis,
    v_grid,
    t_grid,
    nearest_grid,
    polygon,
    half_width: float,
    half_height: float,
    mode: str,
    torch,
) -> None:
    if mode == "x":
        bbox = (torch.abs(u_axis)[None, :, None] <= half_width) & (
            torch.abs(v_grid)[:, None, :] <= half_height
        )
        iz, iy, ix = torch.nonzero(bbox, as_tuple=True)
        if iz.numel() == 0:
            return
        u = u_axis[iy]
        v = v_grid[iz, ix]
        t = t_grid[iz, ix]
        d2 = nearest_grid[iz, ix] + u ** 2
    else:
        bbox = (torch.abs(u_axis)[None, None, :] <= half_width) & (
            torch.abs(v_grid)[:, :, None] <= half_height
        )
        iz, iy, ix = torch.nonzero(bbox, as_tuple=True)
        if iz.numel() == 0:
            return
        u = u_axis[ix]
        v = v_grid[iz, iy]
        t = t_grid[iz, iy]
        d2 = nearest_grid[iz, iy] + u ** 2

    inside = _points_in_polygon_torch(u, v, polygon, torch)
    if not bool(torch.any(inside)):
        return
    iz, iy, ix = iz[inside], iy[inside], ix[inside]
    t = t[inside]
    dist = torch.sqrt(d2[inside]) + torch.abs(t) * 0.1
    update = dist < best_dist[iz, iy, ix]
    if bool(torch.any(update)):
        iz, iy, ix = iz[update], iy[update], ix[update]
        best_dist[iz, iy, ix] = dist[update]
        yarn_id[iz, iy, ix] = int(yarn_index)


def _points_in_polygon_torch(u, v, polygon, torch):
    poly = polygon[:-1]
    p_next = polygon[1:]
    u2 = u[:, None]
    v2 = v[:, None]
    x1 = poly[:, 0]
    y1 = poly[:, 1]
    x2 = p_next[:, 0]
    y2 = p_next[:, 1]
    cond1 = (y1 > v2) != (y2 > v2)
    denom = y2 - y1
    denom = torch.where(torch.abs(denom) < 1e-12, torch.full_like(denom, 1e-12), denom)
    xi = x1 + (v2 - y1) * (x2 - x1) / denom
    hits = (cond1 & (u2 < xi)).sum(dim=-1)
    return (hits % 2) == 1


def _coerce_output(data, output: str, *, device):
    if output == "backend":
        return data
    if output == "numpy":
        return data.to_numpy()
    if output == "torch":
        return data.to_torch(device=device)
    raise ValueError('output must be one of "backend", "numpy", or "torch"')
