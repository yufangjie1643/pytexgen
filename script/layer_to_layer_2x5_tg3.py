"""Build the 2 x 5 layer-to-layer 3D weave from the console log and save TG3.

This script uses TexGen's original Core API.  The pytexgen import is only a
fallback for running the same script from this repository's virtualenv.
"""

import os
import sys


try:
    from _Embedded import *  # noqa: F401,F403
except ImportError:
    pass


def _add_texgen_dll_directories():
    if not hasattr(os, "add_dll_directory"):
        return

    candidates = []
    texgen_root = os.environ.get("TEXGEN_ROOT")
    if texgen_root:
        candidates.append(texgen_root)
        candidates.append(os.path.join(texgen_root, "Python", "libxtra", "TexGen"))

    path_entries = list(sys.path)
    path_entries.extend(os.environ.get("PYTHONPATH", "").split(os.pathsep))
    for path_entry in path_entries:
        if not path_entry:
            continue
        texgen_dir = os.path.abspath(os.path.join(path_entry, "TexGen"))
        if os.path.isfile(os.path.join(texgen_dir, "_Core.pyd")):
            candidates.append(texgen_dir)
            candidates.append(os.path.dirname(os.path.dirname(os.path.dirname(texgen_dir))))

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            try:
                os.add_dll_directory(os.path.abspath(candidate))
            except OSError:
                pass


_add_texgen_dll_directories()

try:
    from TexGen.Core import *  # noqa: F401,F403
except ImportError:
    from pytexgen import *  # noqa: F401,F403


OUTPUT_DIR = "Saved_Layer_To_Layer_2x5"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "layer_to_layer_2x5.tg3")


def create_textile():
    weave = CTextileLayerToLayer(2, 5, 1, 3, 0.1, 0.16, 12)

    weave.SetWarpRatio(0)
    weave.SetBinderRatio(2)
    weave.SetWarpYarnWidths(0.8)
    weave.SetYYarnWidths(1.5)
    weave.SetBinderYarnWidths(1.2)
    weave.SetupLayers(12, 13, 12)
    weave.SetGapSize(0.01)

    binder_positions = [
        (0, 0, 0),
        (1, 0, 1),
        (2, 0, 2),
        (3, 0, 1),
        (4, 0, 0),
        (0, 1, 2),
        (1, 1, 1),
        (2, 1, 0),
        (3, 1, 1),
        (4, 1, 2),
    ]
    for x_index, y_index, layer_index in binder_positions:
        weave.SetBinderPosition(x_index, y_index, layer_index)

    weave.SetWarpYarnPower(0.6)
    weave.SetWeftYarnPower(1)
    weave.SetBinderYarnPower(1)

    for x_index in range(2):
        weave.SetXYarnWidths(x_index, 1.2)
        weave.SetXYarnHeights(x_index, 0.16)
        weave.SetXYarnSpacings(x_index, 1.4)

    for y_index in range(5):
        weave.SetYYarnWidths(y_index, 1.5)
        weave.SetYYarnHeights(y_index, 0.16)
        weave.SetYYarnSpacings(y_index, 3)

    weave.AssignDefaultDomain(False)

    domain = CDomainPlanes()
    domain.AddPlane(PLANE(XYZ(1, 0, 0), 1.5))
    domain.AddPlane(PLANE(XYZ(-1, 0, 0), -13.5))
    domain.AddPlane(PLANE(XYZ(0, 1, 0), 0))
    domain.AddPlane(PLANE(XYZ(0, -1, 0), -12))
    domain.AddPlane(PLANE(XYZ(0, 0, 1), 0.51))
    domain.AddPlane(PLANE(XYZ(0, 0, -1), -3.57))
    weave.AssignDomain(domain)

    return weave


def main():
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    weave = create_textile()
    textile_name = AddTextile(weave)
    SaveToXML(OUTPUT_FILE, textile_name, OUTPUT_STANDARD)

    print("textile_name={0}".format(textile_name))
    print("tg3={0}".format(OUTPUT_FILE))
    return weave, textile_name, OUTPUT_FILE


if __name__ in ("__main__", "<run_path>"):
    main()
