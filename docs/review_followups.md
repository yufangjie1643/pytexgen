# Review Follow-ups

These notes track the review items that were not changed in the P1 test-suite
stabilization pass.

## P2: ACDM custom phase mapping and material LUT mismatch

Location: `TexGen/acdm_solver.py`, around `solve_acdm_isotropic_from_voxel_data`.

`yarn_phase_by_id` can map individual yarn ids to phase ids above `1`, but the
solver path still builds two-entry material lookup tables:

- phase `0`: matrix material
- phase `1`: yarn material

If a caller maps yarns to phase `2`, `3`, or any other valid Voxel-ACDM phase id,
the phase grid and material LUT no longer describe the same material set. The
likely fixes are either:

- restrict `yarn_phase_by_id` to phases already represented by the two-entry LUT,
  and reject other ids with a clear error, or
- add an explicit per-phase material mapping API and build `E_lut` / `nu_lut`
  large enough for the maximum referenced phase.

The second option is more useful for multi-yarn or multi-material workflows.

## P2: ACDM test depends on a sibling Voxel-ACDM checkout

Location: `test_acdm_solver_adapter.py`, `test_find_cloned_voxel_acdm_root`.

The test currently expects `../Voxel-ACDM` to exist next to this repository.
That passes on the current machine, but it is not portable to a clean checkout
or CI runner.

Preferred fix: create a temporary fake Voxel-ACDM root containing a `femlib`
directory and `README.md`, then pass that temp path to `find_voxel_acdm_root`.
That keeps the discovery helper covered without relying on local machine layout.

## P3: Phase id validation happens after uint8 conversion

Location: `TexGen/acdm_solver.py`, `to_acdm_phase_ids`.

The function creates the phase array as `np.uint8` before validating that the
requested ids are in Voxel-ACDM's `0..15` range. With recent NumPy versions,
negative values or values above `255` can raise `OverflowError` during
assignment instead of the intended `ValueError("0..15")`.

Preferred fix: validate `matrix_phase`, `yarn_phase`, and each
`yarn_phase_by_id` value as Python integers before creating or assigning the
`uint8` array. After validation, cast the final phase grid to `np.uint8`.
