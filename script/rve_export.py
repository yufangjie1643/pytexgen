"""Generic layer-aware RVE export helpers for TexGen scripts.

The exporter avoids hard-coded absolute z windows. It first builds an
unclipped probe textile, reads the model z extent and flat yarn centre planes,
then maps layer indices to z-clip windows for one or more RVE mesh exports.

Usage:
    Import export_rve_layers(...) from model scripts. For a ready-to-run
    example, use:

        uv run python script/sic_sic_shallow_cross_straight.py

    That model has default rve_export parameters and can also read JSON from
    @params.json.
"""

from pathlib import Path

try:
    from pytexgen import CRectangularVoxelMesh, DeleteTextile, OUTPUT_STANDARD, SaveToXML, XYZ
except ImportError:
    from TexGen.Core import CRectangularVoxelMesh, DeleteTextile, OUTPUT_STANDARD, SaveToXML, XYZ


def _without_export_keys(params):
    copied = dict(params)
    copied["z_clip_min"] = None
    copied["z_clip_max"] = None
    copied["z_clip_center"] = None
    copied["z_clip_thickness"] = None
    copied["save_tg3"] = False
    copied["mesh_resolutions"] = []
    copied["rve_export"] = None
    return copied


def _normalise_resolution(resolution):
    if isinstance(resolution, (list, tuple)):
        if len(resolution) != 3:
            raise ValueError("mesh resolution tuple must be (x, y, z)")
        return tuple(int(value) for value in resolution)
    value = int(resolution)
    return value, value, value


def _normalise_layers(layers, layer_count, layer_index_base=0):
    if layers is None or str(layers).lower() == "all":
        return list(range(layer_count))
    if isinstance(layers, int):
        layers = [layers]
    normalised = []
    for layer in layers:
        index = int(layer) - int(layer_index_base)
        if index < 0 or index >= layer_count:
            raise ValueError(f"layer index {layer} is outside 0..{layer_count - 1}")
        normalised.append(index)
    return normalised


def _unique_sorted(values, tolerance=1.0e-6):
    unique = []
    for value in sorted(float(item) for item in values):
        if not unique or abs(value - unique[-1]) > tolerance:
            unique.append(value)
    return unique


def _collect_yarn_z_levels(weave, tolerance=1.0e-6):
    flat_centres = []
    node_levels = []
    for yarn_index in range(weave.GetNumYarns()):
        yarn = weave.GetYarn(yarn_index)
        try:
            node_count = yarn.GetNumNodes()
        except AttributeError:
            continue

        z_values = []
        for node_index in range(node_count):
            position = yarn.GetNode(node_index).GetPosition()
            z_values.append(float(position.z))
        if not z_values:
            continue

        node_levels.extend(z_values)
        if max(z_values) - min(z_values) <= tolerance:
            flat_centres.append(sum(z_values) / len(z_values))

    return {
        "flat_yarn_centres": _unique_sorted(flat_centres, tolerance),
        "node_levels": _unique_sorted(node_levels, tolerance),
    }


def _make_layer_windows(z_min, z_max, layer_count, layers, layers_per_rve=1, padding=0.0):
    step = (float(z_max) - float(z_min)) / int(layer_count)
    windows = []
    for start in layers:
        end = min(int(layer_count), int(start) + int(layers_per_rve))
        clip_min = float(z_min) + int(start) * step - float(padding)
        clip_max = float(z_min) + end * step + float(padding)
        clip_min = max(float(z_min), clip_min)
        clip_max = min(float(z_max), clip_max)
        label = f"L{start:02d}" if end == start + 1 else f"L{start:02d}_L{end - 1:02d}"
        windows.append(
            {
                "label": label,
                "layer_start": int(start),
                "layer_end": int(end - 1),
                "z_clip_min": clip_min,
                "z_clip_max": clip_max,
                "z_clip_thickness": clip_max - clip_min,
            }
        )
    return windows


def _make_layer_windows_from_planes(planes, z_min, z_max, layers, layers_per_rve=1, padding=0.0):
    windows = []
    for start in layers:
        end = int(start) + int(layers_per_rve)
        if end >= len(planes):
            raise ValueError(
                f"not enough z planes for layer {start} with layers_per_rve={layers_per_rve}"
            )
        clip_min = float(planes[int(start)]) - float(padding)
        clip_max = float(planes[end]) + float(padding)
        clip_min = max(float(z_min), clip_min)
        clip_max = min(float(z_max), clip_max)
        label = f"L{start:02d}" if end == int(start) + 1 else f"L{start:02d}_L{end - 1:02d}"
        windows.append(
            {
                "label": label,
                "layer_start": int(start),
                "layer_end": int(end - 1),
                "z_clip_min": clip_min,
                "z_clip_max": clip_max,
                "z_clip_thickness": clip_max - clip_min,
            }
        )
    return windows


def probe_model_domain(create_textile, model_params):
    """Build an unclipped model once and return default domain and z levels."""
    probe_params = _without_export_keys(model_params)
    weave, result = create_textile(probe_params)
    try:
        domain = weave.GetDefaultDomain()
        min_point = XYZ()
        max_point = XYZ()
        domain.GetBoxLimits(min_point, max_point)
        z_levels = _collect_yarn_z_levels(weave)
        return {
            "min": (float(min_point.x), float(min_point.y), float(min_point.z)),
            "max": (float(max_point.x), float(max_point.y), float(max_point.z)),
            "textile_name": result.get("textile_name"),
            "flat_yarn_centres": z_levels["flat_yarn_centres"],
            "node_z_levels": z_levels["node_levels"],
        }
    finally:
        textile_name = result.get("textile_name")
        if textile_name:
            DeleteTextile(textile_name)


def resolve_rve_windows(create_textile, model_params, export_config):
    """Resolve configured layer windows against the probed model domain."""
    domain = probe_model_domain(create_textile, model_params)
    layer_count = int(export_config.get("layer_count") or model_params.get("z_layers") or 1)
    if layer_count < 1:
        raise ValueError("RVE layer_count must be >= 1")

    layers = _normalise_layers(
        export_config.get("layers", "all"),
        layer_count,
        export_config.get("layer_index_base", 0),
    )
    layers_per_rve = int(export_config.get("layers_per_rve", 1))
    if layers_per_rve < 1:
        raise ValueError("RVE layers_per_rve must be >= 1")

    z_bounds = export_config.get("z_bounds")
    if z_bounds is None:
        z_min = domain["min"][2]
        z_max = domain["max"][2]
    else:
        if len(z_bounds) != 2:
            raise ValueError("z_bounds must be [z_min, z_max]")
        z_min, z_max = [float(value) for value in z_bounds]
    if z_max <= z_min:
        raise ValueError("RVE z extent must have z_max > z_min")

    window_mode = str(export_config.get("window_mode", "yarn_centres")).lower()
    use_yarn_planes = (
        z_bounds is None
        and window_mode in ("auto", "yarn_centres", "yarn_centers", "flat_yarn_centres", "flat_yarn_centers")
        and len(domain.get("flat_yarn_centres", [])) >= layer_count + 1
    )
    if use_yarn_planes:
        z_planes = domain["flat_yarn_centres"][: layer_count + 1]
        windows = _make_layer_windows_from_planes(
            z_planes,
            z_min,
            z_max,
            layers,
            layers_per_rve=layers_per_rve,
            padding=float(export_config.get("z_padding", 0.0)),
        )
        resolved_mode = "yarn_centres"
    else:
        windows = _make_layer_windows(
            z_min,
            z_max,
            layer_count,
            layers,
            layers_per_rve=layers_per_rve,
            padding=float(export_config.get("z_padding", 0.0)),
        )
        z_planes = None
        resolved_mode = "domain_equal"

    return {
        "domain": domain,
        "layer_count": layer_count,
        "layers_per_rve": layers_per_rve,
        "window_mode": resolved_mode,
        "z_planes": z_planes,
        "windows": windows,
    }


def export_rve_layers(create_textile, model_params, export_config):
    """Export one or more layer windows to TG3 and/or Abaqus INP voxel meshes."""
    if not export_config or not export_config.get("enabled", True):
        return None

    resolved = resolve_rve_windows(create_textile, model_params, export_config)
    save_dir = Path(export_config.get("save_dir") or model_params.get("save_dir") or "RVE_Exports")
    save_dir.mkdir(parents=True, exist_ok=True)

    prefix = export_config.get("file_prefix") or model_params.get("file_prefix") or "rve"
    mesh_resolutions = export_config.get("mesh_resolutions", model_params.get("mesh_resolutions", []))
    mesh_boundary = export_config.get("mesh_boundary", model_params.get("mesh_boundary", "CPeriodicBoundaries"))
    save_tg3 = bool(export_config.get("save_tg3", False))
    voxel_mesh = CRectangularVoxelMesh(mesh_boundary) if mesh_resolutions else None

    exports = []
    for window in resolved["windows"]:
        layer_prefix = f"{prefix}_{window['label']}"
        layer_params = dict(model_params)
        layer_params.update(
            {
                "z_clip_min": window["z_clip_min"],
                "z_clip_max": window["z_clip_max"],
                "z_clip_center": None,
                "z_clip_thickness": None,
                "save_tg3": False,
                "mesh_resolutions": [],
                "rve_export": None,
                "save_dir": str(save_dir),
                "file_prefix": layer_prefix,
            }
        )

        weave, result = create_textile(layer_params)
        try:
            item = dict(window)
            item["textile_name"] = result.get("textile_name")
            item["tg3"] = None
            item["abaqus_meshes"] = []

            if save_tg3:
                tg3_path = save_dir / f"{layer_prefix}.tg3"
                SaveToXML(str(tg3_path), result["textile_name"], OUTPUT_STANDARD)
                item["tg3"] = str(tg3_path)

            for resolution in mesh_resolutions:
                x_res, y_res, z_res = _normalise_resolution(resolution)
                mesh_path = save_dir / f"{layer_prefix}_mesh_{x_res}x{y_res}x{z_res}.inp"
                voxel_mesh.SaveVoxelMesh(
                    weave,
                    str(mesh_path),
                    x_res,
                    y_res,
                    z_res,
                    True,
                    True,
                    5,
                    0,
                )
                item["abaqus_meshes"].append(str(mesh_path))

            exports.append(item)
        finally:
            textile_name = result.get("textile_name")
            if textile_name:
                DeleteTextile(textile_name)

    return {
        "source_domain": resolved["domain"],
        "layer_count": resolved["layer_count"],
        "layers_per_rve": resolved["layers_per_rve"],
        "window_mode": resolved["window_mode"],
        "z_planes": resolved["z_planes"],
        "exports": exports,
    }
