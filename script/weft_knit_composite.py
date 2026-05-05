"""Parametric weft-knit composite TexGen model with central RVE export.

Default run:
    uv run python script/weft_knit_composite.py

Run with JSON overrides:
    uv run python script/weft_knit_composite.py @script/params_weft_knit_composite_rve.json

Preview the generated RVE mesh:
    uv run --with plotly python script/inp_viewer.py Saved_Weft_Knit_Composite/RVE/weft_knit_composite_rve_W03_W04_C03_C04_mesh_64x64x32.inp --backend plotly --output build/weft_knit_rve.html --background white

Notes:
    The parent knit model should be larger than the RVE. By default this script
    builds an 8-by-8 weft-knit parent and exports the central 2-by-2 RVE window.
"""

import copy
import json
import sys
from pathlib import Path


try:
    from _Embedded import *
    from TexGen.Core import *
    from TexGen.Export import *
except ImportError:
    from pytexgen import *


MATERIALS = {
    "fiber_yarn": {
        "description": "effective yarn bundle phase",
        "elastic_modulus_mpa": 70000.0,
        "poisson_ratio": 0.22,
        "density_g_cm3": 1.8,
    },
    "matrix": {
        "description": "matrix phase for voxel mesh background",
        "elastic_modulus_mpa": 3500.0,
        "poisson_ratio": 0.35,
        "density_g_cm3": 1.2,
    },
}


DEFAULT_PARAMS = {
    "wales": 8,
    "courses": 8,
    "wale_height": 1.0,
    "loop_height": 1.2,
    "course_width": 1.0,
    "yarn_thickness": 0.15,
    "assign_default_domain": True,
    "save_dir": "Saved_Weft_Knit_Composite",
    "file_prefix": "weft_knit_composite",
    "save_tg3": True,
    "rve_export": {
        "enabled": True,
        "save_dir": "Saved_Weft_Knit_Composite/RVE",
        "file_prefix": "weft_knit_composite_rve",
        "rve_wales": 2,
        "rve_courses": 2,
        "wale_start": "center",
        "course_start": "center",
        "z_padding": 0.0,
        "save_tg3": True,
        "mesh_boundary": "CPeriodicBoundaries",
        "mesh_resolutions": [(64, 64, 32)],
    },
    "materials": MATERIALS,
}


if "PARAMS" not in globals():
    PARAMS = copy.deepcopy(DEFAULT_PARAMS)


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalise_resolution(resolution):
    if isinstance(resolution, (list, tuple)):
        if len(resolution) != 3:
            raise ValueError("mesh resolution must be [nx, ny, nz]")
        return tuple(int(value) for value in resolution)
    value = int(resolution)
    return value, value, value


def _positive_float(config, key):
    value = float(config[key])
    if value <= 0:
        raise ValueError(f"{key} must be > 0")
    return value


def _positive_int(config, key):
    value = int(config[key])
    if value <= 0:
        raise ValueError(f"{key} must be > 0")
    return value


def _cell_start(value, total, count, name):
    if isinstance(value, str) and value.strip().lower() in ("center", "centre", "central", "middle"):
        return (int(total) - int(count)) // 2
    start = int(value)
    if start < 0 or start + int(count) > int(total):
        raise ValueError(f"{name} start {start} with count {count} is outside 0..{total - 1}")
    return start


def _validate_config(config):
    wales = _positive_int(config, "wales")
    courses = _positive_int(config, "courses")
    _positive_float(config, "wale_height")
    _positive_float(config, "loop_height")
    _positive_float(config, "course_width")
    _positive_float(config, "yarn_thickness")

    rve = config.get("rve_export")
    if rve and rve.get("enabled", True):
        rve_wales = _positive_int(rve, "rve_wales")
        rve_courses = _positive_int(rve, "rve_courses")
        if rve_wales > wales:
            raise ValueError("rve_export.rve_wales must be <= wales")
        if rve_courses > courses:
            raise ValueError("rve_export.rve_courses must be <= courses")
        _cell_start(rve.get("wale_start", "center"), wales, rve_wales, "wale")
        _cell_start(rve.get("course_start", "center"), courses, rve_courses, "course")


def _domain_limits(domain):
    min_point = XYZ()
    max_point = XYZ()
    domain.GetBoxLimits(min_point, max_point)
    return (
        (float(min_point.x), float(min_point.y), float(min_point.z)),
        (float(max_point.x), float(max_point.y), float(max_point.z)),
    )


def _assign_domain(textile, bounds):
    min_values, max_values = bounds
    domain = CDomainPlanes(
        XYZ(float(min_values[0]), float(min_values[1]), float(min_values[2])),
        XYZ(float(max_values[0]), float(max_values[1]), float(max_values[2])),
    )
    textile.AssignDomain(domain)
    return domain


def _make_weft_knit(config, domain_bounds=None):
    textile = CTextileWeftKnit(
        int(config["wales"]),
        int(config["courses"]),
        float(config["wale_height"]),
        float(config["loop_height"]),
        float(config["course_width"]),
        float(config["yarn_thickness"]),
    )
    # CTextileWeftKnit builds its internal yarn lazily; force construction
    # before assigning domains to avoid a SWIG-side access violation.
    textile.GetNumYarns()
    if domain_bounds is None:
        if config.get("assign_default_domain", True):
            textile.AssignDefaultDomain()
    else:
        _assign_domain(textile, domain_bounds)
    return textile


def _resolve_rve_domain(config):
    parent = _make_weft_knit(config)
    parent_domain = parent.GetDefaultDomain()
    domain_min, domain_max = _domain_limits(parent_domain)

    rve = config["rve_export"]
    rve_wales = int(rve["rve_wales"])
    rve_courses = int(rve["rve_courses"])
    wale_start = _cell_start(rve.get("wale_start", "center"), int(config["wales"]), rve_wales, "wale")
    course_start = _cell_start(
        rve.get("course_start", "center"),
        int(config["courses"]),
        rve_courses,
        "course",
    )

    x_min = domain_min[0] + course_start * float(config["course_width"])
    x_max = x_min + rve_courses * float(config["course_width"])
    y_min = domain_min[1] + wale_start * float(config["wale_height"])
    y_max = y_min + rve_wales * float(config["wale_height"])
    z_padding = float(rve.get("z_padding", 0.0))
    z_min = max(domain_min[2], domain_min[2] - z_padding)
    z_max = min(domain_max[2], domain_max[2] + z_padding)

    return {
        "parent_domain": {"min": domain_min, "max": domain_max},
        "bounds": ((x_min, y_min, z_min), (x_max, y_max, z_max)),
        "wale_start": wale_start,
        "wale_end": wale_start + rve_wales - 1,
        "course_start": course_start,
        "course_end": course_start + rve_courses - 1,
        "label": f"W{wale_start:02d}_W{wale_start + rve_wales - 1:02d}_C{course_start:02d}_C{course_start + rve_courses - 1:02d}",
    }


def _save_textile_tg3(textile, textile_name, save_dir, prefix):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tg3_path = save_dir / f"{prefix}.tg3"
    SaveToXML(str(tg3_path), textile_name, OUTPUT_STANDARD)
    return str(tg3_path)


def _export_rve(config):
    rve_config = config.get("rve_export")
    if not rve_config or not rve_config.get("enabled", True):
        return None

    resolved = _resolve_rve_domain(config)
    save_dir = Path(rve_config.get("save_dir") or config.get("save_dir") or "Saved_Weft_Knit_Composite/RVE")
    save_dir.mkdir(parents=True, exist_ok=True)

    prefix = rve_config.get("file_prefix") or f"{config['file_prefix']}_rve"
    layer_prefix = f"{prefix}_{resolved['label']}"
    textile = _make_weft_knit(config, domain_bounds=resolved["bounds"])
    textile_name = AddTextile(textile)
    item = {
        "label": resolved["label"],
        "wale_start": resolved["wale_start"],
        "wale_end": resolved["wale_end"],
        "course_start": resolved["course_start"],
        "course_end": resolved["course_end"],
        "domain": {
            "min": resolved["bounds"][0],
            "max": resolved["bounds"][1],
        },
        "textile_name": textile_name,
        "tg3": None,
        "abaqus_meshes": [],
    }
    try:
        if rve_config.get("save_tg3", True):
            item["tg3"] = _save_textile_tg3(textile, textile_name, save_dir, layer_prefix)

        mesh_resolutions = rve_config.get("mesh_resolutions", [])
        if mesh_resolutions:
            voxel_mesh = CRectangularVoxelMesh(rve_config.get("mesh_boundary", "CPeriodicBoundaries"))
            for resolution in mesh_resolutions:
                nx, ny, nz = _normalise_resolution(resolution)
                mesh_path = save_dir / f"{layer_prefix}_mesh_{nx}x{ny}x{nz}.inp"
                voxel_mesh.SaveVoxelMesh(textile, str(mesh_path), nx, ny, nz, True, True, 5, 0)
                item["abaqus_meshes"].append(str(mesh_path))
    finally:
        DeleteTextile(textile_name)

    return {
        "parent_domain": resolved["parent_domain"],
        "exports": [item],
    }


def create_weft_knit_composite(params=None):
    config = _deep_merge(DEFAULT_PARAMS, params or {})
    _validate_config(config)

    rve_exports = _export_rve(config)
    textile = _make_weft_knit(config)
    textile_name = AddTextile(textile)
    result = {
        "textile_name": textile_name,
        "wales": int(config["wales"]),
        "courses": int(config["courses"]),
        "wale_height": float(config["wale_height"]),
        "loop_height": float(config["loop_height"]),
        "course_width": float(config["course_width"]),
        "yarn_thickness": float(config["yarn_thickness"]),
        "materials": config["materials"],
        "tg3": None,
        "rve_exports": rve_exports,
    }

    if config.get("save_tg3", True):
        result["tg3"] = _save_textile_tg3(
            textile,
            textile_name,
            config.get("save_dir", "Saved_Weft_Knit_Composite"),
            config.get("file_prefix", "weft_knit_composite"),
        )

    return textile, result


def main(params=None):
    global textile, textilename, result

    active_params = copy.deepcopy(PARAMS)
    if params:
        active_params = _deep_merge(active_params, params)
    if __name__ == "__main__" and len(sys.argv) > 1:
        raw_params = sys.argv[1]
        if raw_params.startswith("@"):
            overrides = json.loads(Path(raw_params[1:]).read_text(encoding="utf-8-sig"))
        else:
            overrides = json.loads(raw_params)
        active_params = _deep_merge(active_params, overrides)

    textile, result = create_weft_knit_composite(active_params)
    textilename = result["textile_name"]
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return textile, textilename, result


if __name__ in ("__main__", "<run_path>"):
    main()
