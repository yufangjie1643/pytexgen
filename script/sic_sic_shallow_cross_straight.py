"""SiC/SiC shallow-cross straight layer-to-layer model.

Geometry units are millimetres. Stress units in MATERIALS are MPa.

TexGen embedded Python:

    runpy.run_path(r"E:\\Code\\texgen\\script\\sic_sic_shallow_cross_straight.py")

Current uv environment:

    uv run python script/sic_sic_shallow_cross_straight.py
    uv run python script/sic_sic_shallow_cross_straight.py @params.json

Benchmark default RVE export speed:

    uv run python script/bench_sic_sic_rve_parallel.py
"""

import json
import runpy
import sys
from pathlib import Path


BASE_SCRIPT = Path(__file__).with_name("shallow_cross_layer_to_layer.py")
BASE = runpy.run_path(str(BASE_SCRIPT), run_name="shallow_cross_layer_to_layer_base")
create_shallow_cross_layer_to_layer = BASE["create_shallow_cross_layer_to_layer"]

RVE_SCRIPT = Path(__file__).with_name("rve_export.py")
RVE = runpy.run_path(str(RVE_SCRIPT), run_name="rve_export_base")
export_rve_layers = RVE["export_rve_layers"]


MATERIALS = {
    "sic_fiber_filament": {
        "elastic_modulus_mpa": 373000.0,
        "poisson_ratio": 0.19,
        "radius_mm": 0.006,
        "tensile_strength_mpa": 3600.0,
        "characteristic_tensile_strength_mpa": 3600.0,
        "weibull_shape": [8.0, 15.0],
        "density_g_cm3": 3.0,
    },
    "pyc_interface": {
        "composition": "PyC",
        "thickness_mm": 0.12,
        "normal_elastic_modulus_mpa": [5000.0, 15000.0],
        "tangential_elastic_modulus_mpa": [30000.0, 60000.0],
        "normal_strength_mpa": [10.0, 30.0],
        "shear_strength_mpa": [10.0, 40.0],
        "fracture_energy_j_m2": [5.0, 20.0],
        "density_g_cm3": 2.0,
    },
    "sic_matrix": {
        "elastic_modulus_mpa": [70000.0, 150000.0],
        "poisson_ratio": 0.19,
        "tensile_strength_mpa_lt": 50.0,
        "compressive_strength_mpa": [500.0, 1000.0],
        "fracture_energy_j_m2": [10.0, 30.0],
        "density_g_cm3": [2.5, 2.8],
    },
    "sic_yarn_bundle": {
        "elastic_modulus_mpa": {
            "axial": 242000.0,
            "radial": 220000.0,
        },
        "poisson_ratio": {
            "nu12": 0.17,
            "nu13": 0.17,
            "nu23": 0.20,
        },
        "axial_tensile_strength_mpa": 3950.0,
        "axial_compressive_strength_mpa": 2000.0,
        "in_plane_shear_strength_mpa": 220.0,
        "out_of_plane_shear_strength_mpa": 200.0,
        "bundle_fiber_volume_fraction": [0.35, 0.42],
        "overall_fiber_volume_fraction": [0.33, 0.36],
        "open_porosity_after_densification_lt": 0.05,
    },
}


DEFAULT_PARAMS = {
    "weave_type": "straight",
    "num_x_yarns": 2,
    "num_y_yarns": 4,
    "z_layers": 5,
    "binder_depth": 3,
    # TexGen YYarn is mapped here to the weft yarn group.
    # Keep y_yarn_spacings >= y_yarn_widths to avoid visible overlap.
    "warp_yarn_width": 1.2,
    "weft_yarn_width": 1.5,
    "binder_yarn_width": 0.6,
    "x_yarn_widths": [1.2, 1.2],
    "x_yarn_heights": [0.3, 0.3],
    "x_yarn_spacings": [1.4, 1.4],
    "y_yarn_widths": [1.5, 1.5, 1.5, 1.5],
    "y_yarn_heights": [0.3, 0.3, 0.3, 0.3],
    "y_yarn_spacings": [2.2, 2.2, 2.2, 2.2],
    "x_spacing": 1.4,
    "y_spacing": 2.2,
    "x_height": 0.3,
    "y_height": 0.3,
    "angle_deg": 20.0,
    "section_shape": "power_ellipse",
    "warp_yarn_power": 1.5,
    "weft_yarn_power": 1.5,
    "binder_yarn_power": 1.5,
    "translate": (-1.1, -0.7, 0.0),
    "translate_half_spacing": False,
    "z_clip_min": None,
    "z_clip_max": None,
    "assign_default_domain": True,
    "save_dir": "Saved_SiC_SiC_Shallow_Cross_Straight",
    "file_prefix": "sic_sic_shallow_cross_straight",
    "save_tg3": True,
    # Legacy full-domain mesh exports. Prefer rve_export for layer windows.
    "mesh_resolutions": [],
    "mesh_boundary": "CPeriodicBoundaries",
    "rve_export": {
        "enabled": True,
        "save_dir": "Saved_SiC_SiC_Shallow_Cross_Straight/RVE",
        "file_prefix": "sic_sic_shallow_cross_straight_rve",
        "layer_count": 5,
        "layers": "all",
        "layers_per_rve": 1,
        "layer_index_base": 0,
        "window_mode": "yarn_centres",
        "z_padding": 0.0,
        "save_tg3": True,
        "mesh_boundary": "CPeriodicBoundaries",
        "mesh_resolutions": [(96, 32, 16)],
    },
    "materials": MATERIALS,
}


if "PARAMS" not in globals():
    PARAMS = DEFAULT_PARAMS.copy()


def _validate_yarn_count_multiples(params):
    y_count = int(params["num_x_yarns"])
    x_count = int(params["num_y_yarns"])
    z_count = int(params["z_layers"])

    if z_count < 1:
        raise ValueError("z_layers must be >= 1 and is treated as the z-direction yarn layer count")
    if y_count < 1 or y_count % 2 != 0:
        raise ValueError("num_x_yarns controls the y-direction yarn count and must be a multiple of 2")
    if x_count < 1 or x_count % 4 != 0:
        raise ValueError("num_y_yarns controls the x-direction yarn count and must be a multiple of 4")

    rve_config = params.get("rve_export")
    if rve_config and rve_config.get("enabled", True):
        layer_count = int(rve_config.get("layer_count") or z_count)
        if layer_count < 1:
            raise ValueError("rve_export.layer_count must be >= 1")
        if layer_count != z_count:
            raise ValueError("rve_export.layer_count must match z_layers for SiC/SiC RVE exports")


def create_sic_sic_shallow_cross_straight(params=None):
    active_params = DEFAULT_PARAMS.copy()
    if params:
        active_params.update(params)
    active_params["weave_type"] = "straight"
    _validate_yarn_count_multiples(active_params)

    rve_exports = None
    rve_config = active_params.get("rve_export")
    if rve_config and rve_config.get("enabled", True):
        rve_exports = export_rve_layers(
            create_shallow_cross_layer_to_layer,
            active_params,
            rve_config,
        )

    weave, result = create_shallow_cross_layer_to_layer(active_params)
    if rve_exports:
        result["rve_exports"] = rve_exports

    result["angle_deg"] = active_params["angle_deg"]
    result["materials"] = active_params["materials"]
    return weave, result


def main(params=None):
    global weave, textilename, result

    active_params = PARAMS.copy()
    if params:
        active_params.update(params)
    if __name__ == "__main__" and len(sys.argv) > 1:
        raw_params = sys.argv[1]
        if raw_params.startswith("@"):
            active_params.update(json.loads(Path(raw_params[1:]).read_text(encoding="utf-8-sig")))
        else:
            active_params.update(json.loads(raw_params))

    weave, result = create_sic_sic_shallow_cross_straight(active_params)
    textilename = result["textile_name"]
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return weave, textilename, result


if __name__ in ("__main__", "<run_path>"):
    main()
