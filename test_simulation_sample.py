"""Tests for the framework-neutral simulation sample contract."""

import importlib.util
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parent


def _load_source_module(name):
    path = ROOT / "TexGen" / f"{name.rsplit('.', 1)[-1]}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_simulation_sample():
    """Load source modules without requiring an installed wheel."""
    package = types.ModuleType("TexGen")
    package.__path__ = [str(ROOT / "TexGen")]
    sys.modules["TexGen"] = package

    core = types.ModuleType("TexGen.Core")

    class CYarn:
        LINE = 1
        SURFACE = 2
        VOLUME = 4

    class CTextile:
        pass

    core.CYarn = CYarn
    core.CTextile = CTextile
    sys.modules["TexGen.Core"] = core

    material_fields = _load_source_module("TexGen.material_fields")
    _load_source_module("TexGen.gpu_voxelizer")
    simulation_sample = _load_source_module("TexGen.simulation_sample")
    return simulation_sample, material_fields


class MaterialTableTest(unittest.TestCase):
    def setUp(self):
        self.sample_module, self.material_fields = load_simulation_sample()

    def make_table(self):
        c21 = np.stack(
            (
                self.material_fields.isotropic_stiffness_c21(3.0, 0.30),
                self.material_fields.isotropic_stiffness_c21(70.0, 0.20),
            )
        )
        ids = np.array([0, 7], dtype=np.int32)
        table = self.sample_module.MaterialTable(
            c21=c21,
            material_ids=ids,
            unit="GPa",
            names=("matrix", "carbon"),
        )
        return table, c21, ids

    def test_accepts_explicit_non_dense_ids_without_copy(self):
        table, c21, ids = self.make_table()

        self.assertIs(table.c21, c21)
        self.assertIs(table.material_ids, ids)
        self.assertEqual(table.row_for_id(7), 1)
        self.assertTrue(np.shares_memory(table.c21_for_id(0), c21))
        self.assertEqual(table.names, ("matrix", "carbon"))
        self.assertEqual(table.storage, "numpy")
        self.assertEqual(table.device, "cpu")

    def test_rejects_invalid_identifiers_shapes_and_units(self):
        c21 = np.ones((2, 21), dtype=np.float64)
        MaterialTable = self.sample_module.MaterialTable

        with self.assertRaisesRegex(ValueError, "material ID 0"):
            MaterialTable(c21, np.array([1, 2]), "GPa")
        with self.assertRaisesRegex(ValueError, "unique"):
            MaterialTable(c21, np.array([0, 0]), "GPa")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            MaterialTable(c21, np.array([0, -1]), "GPa")
        with self.assertRaisesRegex(ValueError, r"\(M, 21\)"):
            MaterialTable(np.ones((2, 20)), np.array([0, 1]), "GPa")
        with self.assertRaisesRegex(ValueError, "one-dimensional integer"):
            MaterialTable(c21, np.array([0.0, 1.0]), "GPa")
        with self.assertRaisesRegex(ValueError, "unit"):
            MaterialTable(c21, np.array([0, 1]), " ")
        with self.assertRaisesRegex(ValueError, "names"):
            MaterialTable(c21, np.array([0, 1]), "GPa", names=("matrix",))

    def test_rejects_nonfinite_and_unknown_material(self):
        table, c21, ids = self.make_table()
        c21 = c21.copy()
        c21[1, 3] = np.nan

        with self.assertRaisesRegex(ValueError, "finite"):
            self.sample_module.MaterialTable(c21, ids, "GPa")
        with self.assertRaisesRegex(KeyError, "unknown material ID 99"):
            table.row_for_id(99)

    def test_positive_definite_validation_is_opt_in(self):
        c21 = np.zeros((1, 21), dtype=np.float64)
        ids = np.array([0], dtype=np.int32)

        self.sample_module.MaterialTable(c21, ids, "Pa")
        with self.assertRaisesRegex(ValueError, "positive definite"):
            self.sample_module.MaterialTable(
                c21,
                ids,
                "Pa",
                validate_positive_definite=True,
            )

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_conversion_preserves_ids_and_copy_semantics(self):
        table, c21, ids = self.make_table()

        same = table.to(copy=False)
        copied = table.to(copy=True)
        torch_table = table.to("torch", dtype=torch.float32)

        self.assertIs(same.c21, c21)
        self.assertIs(same.material_ids, ids)
        self.assertIsNot(copied.c21, c21)
        self.assertIsNot(copied.material_ids, ids)
        np.testing.assert_array_equal(copied.material_ids, ids)
        self.assertEqual(torch_table.c21.dtype, torch.float32)
        self.assertEqual(torch_table.material_ids.dtype, torch.int32)
        self.assertEqual(torch_table.device, "cpu")
        np.testing.assert_array_equal(torch_table.material_ids.numpy(), ids)

        same_torch = torch_table.to("torch", copy=False)
        copied_torch = torch_table.to("torch", copy=True)
        self.assertIs(same_torch.c21, torch_table.c21)
        self.assertEqual(
            same_torch.material_ids.data_ptr(),
            torch_table.material_ids.data_ptr(),
        )
        self.assertNotEqual(
            copied_torch.c21.data_ptr(),
            torch_table.c21.data_ptr(),
        )

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_rejects_mixed_backends_and_numpy_cuda_target(self):
        c21 = np.ones((2, 21), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "same storage backend"):
            self.sample_module.MaterialTable(
                c21,
                torch.tensor([0, 1], dtype=torch.int32),
                "GPa",
            )

        table, _, _ = self.make_table()
        with self.assertRaisesRegex(ValueError, "NumPy.*CPU"):
            table.to("numpy", device="cuda")


class SimulationSampleValidationTest(unittest.TestCase):
    def setUp(self):
        self.sample_module, self.material_fields = load_simulation_sample()
        self.voxelizer = sys.modules["TexGen.gpu_voxelizer"]

    def make_components(self):
        indices = np.array([1, 2], dtype=np.int64)
        yarn_ids = np.array([0, 2], dtype=np.int32)
        orientation = self.material_fields.SparseOrientationField(
            voxel_indices=indices,
            yarn_ids=yarn_ids,
            orientation1=np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            orientation2=np.array(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            grid_shape=(1, 2, 2),
        )
        matrix = self.material_fields.isotropic_stiffness_c21(3.0, 0.30)
        material_7 = self.material_fields.isotropic_stiffness_c21(70.0, 0.20)
        material_9 = self.material_fields.isotropic_stiffness_c21(120.0, 0.25)
        materials = self.sample_module.MaterialTable(
            c21=np.stack((matrix, material_7, material_9)),
            material_ids=np.array([0, 7, 9], dtype=np.int32),
            unit="GPa",
            names=("matrix", "yarn-a", "yarn-b"),
        )
        stiffness = self.material_fields.SparseStiffnessField(
            matrix_c21=materials.c21[0],
            voxel_indices=indices,
            yarn_ids=yarn_ids,
            material_ids=np.array([7, 9], dtype=np.int32),
            yarn_c21=np.stack((material_7, material_9)),
            grid_shape=(1, 2, 2),
            unit="GPa",
        )
        voxels = self.voxelizer.VoxelGridData(
            yarn_id=np.array([-1, 0, 2, -1], dtype=np.int32),
            aabb=np.array(
                [[0.0, 0.0, 0.0], [2.0, 2.0, 1.0]],
                dtype=np.float64,
            ),
            resolution=(2, 2, 1),
            backend="numpy",
            device="cpu",
            workers=1,
            dtype="float64",
            timings={"classify": 0.01},
            sparse_orientation=orientation,
            storage="numpy",
        )
        return voxels, orientation, stiffness, materials

    def make_sample(self, metadata=None):
        voxels, orientation, stiffness, materials = self.make_components()
        return self.sample_module.SimulationSample(
            voxels=voxels,
            orientation=orientation,
            stiffness=stiffness,
            materials=materials,
            metadata=(
                {
                    "source": "unit-test",
                    "generation": {"seed": 3},
                    "tags": ["gpu", "training"],
                }
                if metadata is None
                else metadata
            ),
        )

    def test_construction_is_zero_copy_and_adopts_voxel_orientation(self):
        voxels, orientation, stiffness, materials = self.make_components()

        sample = self.sample_module.SimulationSample(
            voxels=voxels,
            stiffness=stiffness,
            materials=materials,
            metadata={"source": "unit-test"},
        )

        self.assertIs(sample.voxels, voxels)
        self.assertIs(sample.orientation, orientation)
        self.assertIs(sample.stiffness, stiffness)
        self.assertIs(sample.materials, materials)
        self.assertEqual(sample.storage, "numpy")
        self.assertEqual(sample.device, "cpu")

    def test_metadata_is_detached_json_compatible_and_recursively_immutable(self):
        metadata = {
            "source": "unit-test",
            "generation": {"seed": 3},
            "tags": ["gpu", "training"],
        }
        sample = self.make_sample(metadata)
        metadata["generation"]["seed"] = 99
        metadata["tags"].append("mutated")

        self.assertEqual(sample.metadata["generation"]["seed"], 3)
        self.assertEqual(sample.metadata["tags"], ("gpu", "training"))
        with self.assertRaises(TypeError):
            sample.metadata["generation"]["seed"] = 4
        with self.assertRaisesRegex(ValueError, "JSON-compatible"):
            self.make_sample({"bad": np.array([1, 2])})

    def test_resident_field_registry_returns_original_arrays(self):
        sample = self.make_sample()

        self.assertTrue(
            np.shares_memory(
                sample.array("voxel.yarn_id"),
                sample.voxels.yarn_id,
            )
        )
        self.assertIs(
            sample.array("orientation.primary"),
            sample.orientation.orientation1,
        )
        self.assertIs(
            sample.array("orientation.secondary"),
            sample.orientation.orientation2,
        )
        self.assertIs(
            sample.array("stiffness.yarn_c21"),
            sample.stiffness.yarn_c21,
        )
        self.assertIs(
            sample.array("stiffness.material_ids"),
            sample.stiffness.material_ids,
        )
        self.assertIs(sample.array("material.c21"), sample.materials.c21)
        self.assertIs(
            sample.array("material.ids"),
            sample.materials.material_ids,
        )
        self.assertIn("voxel.occupancy", sample.field_names)
        self.assertIn("voxel.material_id", sample.field_names)

    def test_rejects_conflicting_voxel_orientation_owner(self):
        voxels, orientation, stiffness, materials = self.make_components()
        duplicate = replace(
            orientation,
            voxel_indices=orientation.voxel_indices.copy(),
        )

        with self.assertRaisesRegex(ValueError, "same object"):
            self.sample_module.SimulationSample(
                voxels=voxels,
                orientation=duplicate,
                stiffness=stiffness,
                materials=materials,
            )

    def test_rejects_shape_order_and_sparse_identity_mismatches(self):
        voxels, orientation, stiffness, materials = self.make_components()

        with self.assertRaisesRegex(ValueError, "grid shape"):
            self.sample_module.SimulationSample(
                voxels=voxels,
                orientation=orientation,
                stiffness=replace(stiffness, grid_shape=(1, 1, 4)),
                materials=materials,
            )
        with self.assertRaisesRegex(ValueError, "voxel order"):
            self.sample_module.SimulationSample(
                voxels=voxels,
                orientation=orientation,
                stiffness=replace(stiffness, order="different"),
                materials=materials,
            )
        with self.assertRaisesRegex(ValueError, "voxel indices"):
            self.sample_module.SimulationSample(
                voxels=voxels,
                orientation=orientation,
                stiffness=replace(
                    stiffness,
                    voxel_indices=np.array([1, 3], dtype=np.int64),
                ),
                materials=materials,
            )
        with self.assertRaisesRegex(ValueError, "yarn IDs"):
            self.sample_module.SimulationSample(
                voxels=voxels,
                orientation=orientation,
                stiffness=replace(
                    stiffness,
                    yarn_ids=np.array([0, 3], dtype=np.int32),
                ),
                materials=materials,
            )

    def test_rejects_material_identity_matrix_and_unit_mismatches(self):
        voxels, orientation, stiffness, materials = self.make_components()

        with self.assertRaisesRegex(ValueError, "unknown material ID 11"):
            self.sample_module.SimulationSample(
                voxels=voxels,
                orientation=orientation,
                stiffness=replace(
                    stiffness,
                    material_ids=np.array([7, 11], dtype=np.int32),
                ),
                materials=materials,
            )
        with self.assertRaisesRegex(ValueError, "matrix stiffness"):
            self.sample_module.SimulationSample(
                voxels=voxels,
                orientation=orientation,
                stiffness=replace(
                    stiffness,
                    matrix_c21=2.0 * stiffness.matrix_c21,
                ),
                materials=materials,
            )
        with self.assertRaisesRegex(ValueError, "unit"):
            self.sample_module.SimulationSample(
                voxels=voxels,
                orientation=orientation,
                stiffness=replace(stiffness, unit="Pa"),
                materials=materials,
            )

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_rejects_storage_backend_mismatch(self):
        voxels, orientation, stiffness, materials = self.make_components()

        with self.assertRaisesRegex(ValueError, "storage backend"):
            self.sample_module.SimulationSample(
                voxels=voxels,
                orientation=orientation,
                stiffness=stiffness,
                materials=materials.to("torch"),
            )


class SimulationSampleFieldTest(unittest.TestCase):
    def setUp(self):
        fixture = SimulationSampleValidationTest(
            "test_construction_is_zero_copy_and_adopts_voxel_orientation"
        )
        fixture.setUp()
        self.fixture = fixture
        self.sample_module = fixture.sample_module
        self.sample = fixture.make_sample()

    def test_derived_fields_require_explicit_copy_and_use_physical_ids(self):
        with self.assertRaisesRegex(ValueError, "copy=True"):
            self.sample.array("voxel.occupancy")
        with self.assertRaisesRegex(ValueError, "copy=True"):
            self.sample.array("voxel.material_id")

        occupancy = self.sample.array("voxel.occupancy", copy=True)
        material_grid = self.sample.array("voxel.material_id", copy=True)

        np.testing.assert_array_equal(
            occupancy,
            np.array([[[False, True], [True, False]]]),
        )
        np.testing.assert_array_equal(
            material_grid,
            np.array([[[0, 7], [9, 0]]], dtype=np.int32),
        )
        self.assertFalse(
            np.array_equal(material_grid, self.sample.voxels.material_id())
        )

    def test_acdm_layout_requires_copy_and_matches_sparse_reference(self):
        with self.assertRaisesRegex(ValueError, "copy=True"):
            self.sample.array(
                "stiffness.yarn_c21",
                layout="acdm",
            )

        dense = self.sample.array(
            "stiffness.yarn_c21",
            layout="acdm",
            copy=True,
        )

        self.assertEqual(dense.shape, (1, 6, 6, 1, 2, 2))
        np.testing.assert_allclose(
            dense,
            self.sample.stiffness.to_acdm(batch=True),
        )

    def test_native_copy_and_dictionary_semantics_are_explicit(self):
        resident = self.sample.array("material.c21")
        copied = self.sample.array("material.c21", copy=True)
        zero_copy_dict = self.sample.as_dict(copy=False)
        complete_dict = self.sample.as_dict(copy=True)

        self.assertIs(resident, self.sample.materials.c21)
        self.assertIsNot(copied, resident)
        np.testing.assert_array_equal(copied, resident)
        self.assertNotIn("voxel.occupancy", zero_copy_dict)
        self.assertNotIn("voxel.material_id", zero_copy_dict)
        self.assertIn("voxel.occupancy", complete_dict)
        self.assertIn("voxel.material_id", complete_dict)
        self.assertIs(
            zero_copy_dict["orientation.voxel_indices"],
            self.sample.orientation.voxel_indices,
        )

    def test_unknown_unavailable_and_unsupported_layouts_are_clear(self):
        with self.assertRaisesRegex(KeyError, "available fields"):
            self.sample.array("unknown.field")
        with self.assertRaisesRegex(ValueError, "not supported"):
            self.sample.array("material.c21", layout="channels_first")

        voxels = replace(self.sample.voxels, sparse_orientation=None)
        matrix_only = self.sample_module.SimulationSample(
            voxels=voxels,
            materials=self.sample.materials,
        )
        self.assertNotIn("voxel.material_id", matrix_only.field_names)
        self.assertNotIn("orientation.primary", matrix_only.field_names)
        with self.assertRaisesRegex(KeyError, "unavailable"):
            matrix_only.array("voxel.material_id", copy=True)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_to_converts_all_arrays_and_reuses_one_sparse_topology(self):
        torch_sample = self.sample.to("torch", dtype=torch.float32)

        self.assertEqual(torch_sample.storage, "torch")
        self.assertEqual(torch_sample.device, "cpu")
        self.assertEqual(torch_sample.materials.c21.dtype, torch.float32)
        self.assertEqual(
            torch_sample.orientation.orientation1.dtype,
            torch.float32,
        )
        self.assertEqual(torch_sample.stiffness.yarn_c21.dtype, torch.float32)
        self.assertEqual(
            torch_sample.materials.material_ids.dtype,
            torch.int32,
        )
        self.assertIs(
            torch_sample.orientation,
            torch_sample.voxels.sparse_orientation,
        )
        self.assertIs(
            torch_sample.stiffness.voxel_indices,
            torch_sample.orientation.voxel_indices,
        )
        self.assertIs(
            torch_sample.stiffness.yarn_ids,
            torch_sample.orientation.yarn_ids,
        )
        self.assertEqual(torch_sample.metadata, self.sample.metadata)

        restored = torch_sample.to("numpy", dtype=np.float64)
        np.testing.assert_array_equal(
            restored.stiffness.material_ids,
            self.sample.stiffness.material_ids,
        )
        np.testing.assert_allclose(
            restored.stiffness.yarn_c21,
            self.sample.stiffness.yarn_c21,
        )

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_to_returns_self_for_identity_and_clones_when_requested(self):
        torch_sample = self.sample.to("torch", dtype=torch.float64)

        self.assertIs(self.sample.to(copy=False), self.sample)
        self.assertIs(
            torch_sample.to(
                "torch",
                device="cpu",
                dtype=torch.float64,
                copy=False,
            ),
            torch_sample,
        )
        copied = torch_sample.to(copy=True)
        self.assertIsNot(copied, torch_sample)
        self.assertNotEqual(
            copied.materials.c21.data_ptr(),
            torch_sample.materials.c21.data_ptr(),
        )
        self.assertNotEqual(
            copied.voxels.yarn_id.data_ptr(),
            torch_sample.voxels.yarn_id.data_ptr(),
        )


class SimulationSampleWorkflowTest(unittest.TestCase):
    def setUp(self):
        fixture = SimulationSampleValidationTest(
            "test_construction_is_zero_copy_and_adopts_voxel_orientation"
        )
        fixture.setUp()
        self.fixture = fixture
        self.sample_module = fixture.sample_module

    def test_one_call_resolves_explicit_material_rows(self):
        voxels, orientation, stiffness, materials = (
            self.fixture.make_components()
        )
        textile = object()

        with mock.patch.object(
            self.sample_module,
            "voxelize_textile_material_fields",
            return_value=(voxels, stiffness),
        ) as voxelize:
            sample = self.sample_module.voxelize_textile_simulation_sample(
                textile,
                materials=materials,
                default_yarn_material_id=7,
                yarn_material_id_by_id={2: 9},
                metadata={"case": "workflow"},
                nx=2,
                ny=2,
                nz=1,
                backend="numpy",
            )

        self.assertIsInstance(sample, self.sample_module.SimulationSample)
        self.assertIs(sample.voxels, voxels)
        self.assertIs(sample.stiffness, stiffness)
        self.assertEqual(sample.metadata["case"], "workflow")
        kwargs = voxelize.call_args.kwargs
        np.testing.assert_array_equal(
            kwargs["matrix_stiffness"],
            materials.c21_for_id(0),
        )
        np.testing.assert_array_equal(
            kwargs["default_yarn_stiffness"],
            materials.c21_for_id(7),
        )
        np.testing.assert_array_equal(
            kwargs["yarn_stiffness_by_id"][2],
            materials.c21_for_id(9),
        )
        self.assertEqual(kwargs["default_yarn_material_id"], 7)
        self.assertEqual(kwargs["yarn_material_id_by_id"], {2: 9})
        self.assertEqual(kwargs["unit"], "GPa")
        self.assertEqual(kwargs["nx"], 2)

    def test_one_call_rejects_unknown_explicit_material_id(self):
        _, _, _, materials = self.fixture.make_components()

        with self.assertRaisesRegex(KeyError, "unknown material ID 99"):
            self.sample_module.voxelize_textile_simulation_sample(
                object(),
                materials=materials,
                default_yarn_material_id=99,
                nx=2,
                ny=2,
                nz=1,
            )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SimulationSampleDLPackTest(unittest.TestCase):
    def setUp(self):
        fixture = SimulationSampleValidationTest(
            "test_construction_is_zero_copy_and_adopts_voxel_orientation"
        )
        fixture.setUp()
        self.sample = fixture.make_sample().to(
            "torch",
            dtype=torch.float32,
        )

    def test_cpu_fields_share_the_original_allocation(self):
        self.assertFalse(hasattr(self.sample, "__dlpack__"))

        for name, field in self.sample.as_dict(copy=False).items():
            with self.subTest(field=name):
                shared = torch.from_dlpack(field)
                self.assertEqual(shared.data_ptr(), field.data_ptr())
                self.assertEqual(tuple(shared.shape), tuple(field.shape))

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_cuda_fields_share_the_original_allocation(self):
        cuda_sample = self.sample.to("torch", device="cuda")

        for name, field in cuda_sample.as_dict(copy=False).items():
            with self.subTest(field=name):
                shared = torch.from_dlpack(field)
                self.assertTrue(shared.is_cuda)
                self.assertEqual(shared.data_ptr(), field.data_ptr())
                self.assertEqual(shared.device, field.device)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_nondefault_cuda_stream_observes_producer_write(self):
        cuda_sample = self.sample.to("torch", device="cuda")
        field = cuda_sample.array("stiffness.yarn_c21")
        expected = torch.full_like(field, 37.0)
        producer = torch.cuda.Stream(device=field.device)
        consumer = torch.cuda.Stream(device=field.device)

        with torch.cuda.stream(producer):
            field.copy_(expected)
        with torch.cuda.stream(consumer):
            shared = torch.from_dlpack(field)
            observed = shared.clone()
        consumer.synchronize()

        torch.testing.assert_close(observed, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
