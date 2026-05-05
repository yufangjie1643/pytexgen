"""Parametric shallow-cross layer-to-layer TexGen model.

Supported weave types:

    bent      / 浅交弯联
    straight  / 浅交直联

TexGen embedded Python example:

    runpy.run_path(
        r"E:\\Code\\texgen\\script\\shallow_cross_layer_to_layer.py",
        init_globals={"PARAMS": {"weave_type": "straight", "z_layers": 5}},
    )

Current uv environment example:

    uv run python script/shallow_cross_layer_to_layer.py '{"weave_type":"bent"}'
    uv run python script/shallow_cross_layer_to_layer.py @params.json
"""

import json
import math
import runpy
import sys
from pathlib import Path


try:
    from _Embedded import *
    from TexGen.Core import *
    from TexGen.Renderer import *
    from TexGen.Export import *
    from TexGen.WeavePattern import *
    from TexGen.WiseTex import *
    from TexGen.FlowTex import *
except ImportError:
    from pytexgen import *


WEAVE_TYPE_ALIASES = {
    "bent": "bent",
    "bend": "bent",
    "curved": "bent",
    "shallow_cross_binder": "bent",
    "浅交弯联": "bent",
    "straight": "straight",
    "direct": "straight",
    "shallow_cross_straight": "straight",
    "shallow_cross_straight_binder": "straight",
    "浅交直联": "straight",
}


DEFAULT_PARAMS = {
    "weave_type": "bent",
    "num_x_yarns": 2,
    "num_y_yarns": 4,
    "x_spacing": 1.0,
    "y_spacing": 1.0,
    "x_height": 0.1,
    "y_height": 0.1,
    "z_layers": None,
    "warp_layers": None,
    "weft_layers": None,
    "binder_layers": None,
    "num_binder_layers": None,
    "binder_depth": None,
    "straight_binder_depth": None,
    "warp_ratio": 0,
    "binder_ratio": 2,
    "warp_yarn_width": 0.8,
    "warp_yarn_height": None,
    "weft_yarn_width": 0.8,
    "weft_yarn_height": None,
    "binder_yarn_width": 0.4,
    "binder_yarn_height": None,
    "x_yarn_widths": 0.4,
    "x_yarn_heights": 0.05,
    "x_yarn_spacings": 0.5,
    "y_yarn_widths": 0.8,
    "y_yarn_heights": 0.1,
    "y_yarn_spacings": 1.0,
    "gap_size": 0.0,
    "binder_positions": None,
    "section_shape": "power_ellipse",
    "warp_yarn_power": 0.6,
    "weft_yarn_power": 0.6,
    "binder_yarn_power": 0.6,
    "explicit_section_assignment": None,
    "section_width": None,
    "section_height": None,
    "section_mesh_layers": None,
    "translate": None,
    "translate_half_spacing": False,
    "z_clip_min": None,
    "z_clip_max": None,
    "z_clip_center": None,
    "z_clip_thickness": None,
    "assign_default_domain": True,
    "save_dir": "Saved_Shallow_Cross_Textiles",
    "file_prefix": None,
    "save_tg3": False,
}


if "PARAMS" not in globals():
    PARAMS = DEFAULT_PARAMS.copy()


def _normalise_weave_type(value):
    key = str(value).strip().lower()
    if value in WEAVE_TYPE_ALIASES:
        return WEAVE_TYPE_ALIASES[value]
    if key in WEAVE_TYPE_ALIASES:
        return WEAVE_TYPE_ALIASES[key]
    raise ValueError("weave_type must be 'bent'/'浅交弯联' or 'straight'/'浅交直联'")


def _as_sequence(value, count, name):
    if isinstance(value, (int, float)):
        return [float(value)] * count
    if len(value) != count:
        raise ValueError(f"{name} must contain {count} values")
    return [float(item) for item in value]


def _set_yarn_series(weave, setter_name, values, count, name):
    setter = getattr(weave, setter_name)
    for index, value in enumerate(_as_sequence(values, count, name)):
        setter(index, value)


def _merge_params(params):
    cfg = DEFAULT_PARAMS.copy()
    if params:
        cfg.update(params)

    weave_type = _normalise_weave_type(cfg["weave_type"])
    cfg["weave_type"] = weave_type

    if cfg["z_layers"] is None:
        cfg["z_layers"] = 3 if weave_type == "bent" else 5
    cfg["z_layers"] = int(cfg["z_layers"])
    if cfg["z_layers"] < 1:
        raise ValueError("z_layers must be >= 1")

    cfg["num_binder_layers"] = int(cfg["num_binder_layers"] or cfg["z_layers"])
    cfg["warp_layers"] = int(cfg["warp_layers"] or cfg["z_layers"])
    cfg["weft_layers"] = int(cfg["weft_layers"] or (cfg["z_layers"] + 1))
    cfg["binder_layers"] = int(cfg["binder_layers"] or cfg["z_layers"])

    if cfg["warp_yarn_height"] is None:
        cfg["warp_yarn_height"] = cfg["x_yarn_heights"]
    if cfg["weft_yarn_height"] is None:
        cfg["weft_yarn_height"] = cfg["y_yarn_heights"]
    if cfg["binder_yarn_height"] is None:
        cfg["binder_yarn_height"] = cfg["x_yarn_heights"]

    if cfg["file_prefix"] is None:
        cfg["file_prefix"] = f"shallow_cross_{weave_type}_binder"

    return cfg


def auto_binder_positions(
    weave_type,
    num_x_yarns,
    num_y_yarns,
    z_layers,
    binder_depth=None,
    straight_binder_depth=None,
):
    """Generate SetBinderPosition(x, y, zOffset) tuples for the selected weave."""
    weave_type = _normalise_weave_type(weave_type)
    num_x_yarns = int(num_x_yarns)
    num_y_yarns = int(num_y_yarns)
    z_layers = int(z_layers)
    if binder_depth is None:
        binder_depth = min(3, z_layers)
    binder_depth = int(binder_depth)
    if binder_depth < 1 or binder_depth > z_layers:
        raise ValueError("binder_depth must be between 1 and z_layers")

    if weave_type == "bent":
        max_offset = binder_depth - 1
        return [
            (x_index, y_index, 0 if (x_index + y_index) % 2 == 0 else max_offset)
            for y_index in range(num_x_yarns)
            for x_index in range(num_y_yarns)
        ]

    if straight_binder_depth is None:
        straight_binder_depth = binder_depth
    peak = max(0, int(straight_binder_depth) - 1)
    if peak >= z_layers:
        raise ValueError("straight_binder_depth must be <= z_layers")

    positions = []
    period = max(1, 2 * peak)
    for y_index in range(num_x_yarns):
        for x_index in range(num_y_yarns):
            if peak == 0:
                offset = 0
            else:
                phase = x_index % period
                offset = phase if phase <= peak else period - phase
                if y_index % 2:
                    offset = peak - offset
            positions.append((x_index, y_index, offset))
    return positions


def _create_section(shape, width, height, power):
    shape = str(shape).strip().lower()
    if shape in ("power_ellipse", "powerellipse", "superellipse"):
        section = CSectionPowerEllipse(float(width), float(height), float(power))
    elif shape in ("ellipse", "elliptical"):
        section = CSectionEllipse(float(width), float(height))
    elif shape in ("lenticular", "lens"):
        section = CSectionLenticular(float(width), float(height))
    elif shape in ("rectangle", "rect", "矩形"):
        section = CSectionRectangle(float(width), float(height))
    else:
        raise ValueError("section_shape must be power_ellipse, ellipse, lenticular, or rectangle")
    return section


def _average(values):
    if isinstance(values, (int, float)):
        return float(values)
    return sum(float(value) for value in values) / len(values)


def _assign_explicit_sections(weave, cfg):
    width = cfg["section_width"]
    height = cfg["section_height"]
    if width is None:
        width = _average([cfg["warp_yarn_width"], cfg["weft_yarn_width"], cfg["binder_yarn_width"]])
    if height is None:
        height = _average([_average(cfg["x_yarn_heights"]), _average(cfg["y_yarn_heights"])])

    section = _create_section(cfg["section_shape"], width, height, cfg["warp_yarn_power"])
    if cfg["section_mesh_layers"] is not None:
        section.SetSectionMeshLayers(int(cfg["section_mesh_layers"]))
    yarn_section = CYarnSectionConstant(section)

    for index in range(weave.GetNumYarns()):
        weave.GetYarn(index).AssignSection(yarn_section)


def create_shallow_cross_layer_to_layer(params=None):
    """Create a shallow-cross layer-to-layer textile and optionally save it."""
    cfg = _merge_params(params)
    num_x_yarns = int(cfg["num_x_yarns"])
    num_y_yarns = int(cfg["num_y_yarns"])

    weave = CTextileLayerToLayer(
        num_x_yarns,
        num_y_yarns,
        float(cfg["x_spacing"]),
        float(cfg["y_spacing"]),
        float(cfg["x_height"]),
        float(cfg["y_height"]),
        int(cfg["num_binder_layers"]),
    )

    weave.SetWarpRatio(int(cfg["warp_ratio"]))
    weave.SetBinderRatio(int(cfg["binder_ratio"]))
    weave.SetWarpYarnWidths(float(cfg["warp_yarn_width"]))
    weave.SetYYarnWidths(float(cfg["weft_yarn_width"]))
    weave.SetBinderYarnWidths(float(cfg["binder_yarn_width"]))
    if cfg["warp_yarn_height"] is not None:
        weave.SetWarpYarnHeights(float(_average(cfg["warp_yarn_height"])))
    if cfg["binder_yarn_height"] is not None:
        weave.SetBinderYarnHeights(float(_average(cfg["binder_yarn_height"])))

    weave.SetupLayers(
        int(cfg["warp_layers"]),
        int(cfg["weft_layers"]),
        int(cfg["binder_layers"]),
    )
    weave.SetGapSize(float(cfg["gap_size"]))

    positions = cfg["binder_positions"]
    if positions is None:
        positions = auto_binder_positions(
            cfg["weave_type"],
            num_x_yarns,
            num_y_yarns,
            cfg["z_layers"],
            cfg["binder_depth"],
            cfg["straight_binder_depth"],
        )
    for x_index, y_index, layer_index in positions:
        weave.SetBinderPosition(int(x_index), int(y_index), int(layer_index))

    weave.SetWarpYarnPower(float(cfg["warp_yarn_power"]))
    weave.SetWeftYarnPower(float(cfg["weft_yarn_power"]))
    weave.SetBinderYarnPower(float(cfg["binder_yarn_power"]))

    _set_yarn_series(weave, "SetXYarnWidths", cfg["x_yarn_widths"], num_x_yarns, "x_yarn_widths")
    _set_yarn_series(weave, "SetXYarnHeights", cfg["x_yarn_heights"], num_x_yarns, "x_yarn_heights")
    _set_yarn_series(weave, "SetXYarnSpacings", cfg["x_yarn_spacings"], num_x_yarns, "x_yarn_spacings")
    _set_yarn_series(weave, "SetYYarnWidths", cfg["y_yarn_widths"], num_y_yarns, "y_yarn_widths")
    _set_yarn_series(weave, "SetYYarnHeights", cfg["y_yarn_heights"], num_y_yarns, "y_yarn_heights")
    _set_yarn_series(weave, "SetYYarnSpacings", cfg["y_yarn_spacings"], num_y_yarns, "y_yarn_spacings")

    explicit_sections = cfg["explicit_section_assignment"]
    if explicit_sections is None:
        explicit_sections = str(cfg["section_shape"]).strip().lower() not in (
            "power_ellipse",
            "powerellipse",
            "superellipse",
        )
    if explicit_sections:
        _assign_explicit_sections(weave, cfg)

    z_clip_min = cfg["z_clip_min"]
    z_clip_max = cfg["z_clip_max"]

    if cfg["assign_default_domain"]:
        domain = weave.GetDefaultDomain()
        if cfg["z_clip_thickness"] is not None:
            z_center = cfg["z_clip_center"]
            if z_center is None:
                default_min = XYZ()
                default_max = XYZ()
                domain.GetBoxLimits(default_min, default_max)
                z_center = (default_min.z + default_max.z) / 2.0
            half_thickness = float(cfg["z_clip_thickness"]) / 2.0
            z_clip_min = float(z_center) - half_thickness
            z_clip_max = float(z_center) + half_thickness
        if z_clip_min is not None or z_clip_max is not None:
            default_min = XYZ()
            default_max = XYZ()
            domain.GetBoxLimits(default_min, default_max)
            if z_clip_min is None:
                z_clip_min = default_min.z
            if z_clip_max is None:
                z_clip_max = default_max.z
            domain = CDomainPlanes(
                XYZ(default_min.x, default_min.y, float(z_clip_min)),
                XYZ(default_max.x, default_max.y, float(z_clip_max)),
            )
        weave.AssignDomain(domain)

    translate = cfg["translate"]
    if cfg["translate_half_spacing"]:
        translate = (
            -float(cfg["x_spacing"]) / 2.0,
            -float(cfg["y_spacing"]) / 2.0,
            0.0,
        )
    if translate is not None:
        weave.Translate(XYZ(float(translate[0]), float(translate[1]), float(translate[2])))
        translate = (float(translate[0]), float(translate[1]), float(translate[2]))

    textile_name = AddTextile(weave)
    result = {
        "textile_name": textile_name,
        "weave_type": cfg["weave_type"],
        "z_layers": cfg["z_layers"],
        "binder_positions": positions,
        "translate": translate,
        "z_clip": (
            None
            if cfg["z_clip_min"] is None
            and cfg["z_clip_max"] is None
            and cfg["z_clip_thickness"] is None
            else {
                "min": z_clip_min,
                "max": z_clip_max,
                "thickness": None if z_clip_min is None or z_clip_max is None else z_clip_max - z_clip_min,
            }
        ),
    }

    if cfg["save_tg3"]:
        save_dir = Path(cfg["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        tg3_path = save_dir / f"{cfg['file_prefix']}.tg3"
        SaveToXML(str(tg3_path), textile_name, OUTPUT_STANDARD)
        result["tg3"] = str(tg3_path)

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

    weave, result = create_shallow_cross_layer_to_layer(active_params)
    textilename = result["textile_name"]
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return weave, textilename, result


if __name__ in ("__main__", "<run_path>"):
    main()
