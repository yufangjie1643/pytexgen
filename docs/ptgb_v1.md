# PTGB Prepared Geometry Format, Version 1

PTGB is PyTexGen's read-only, memory-mappable geometry cache for repeated
voxelization. TG3 remains the canonical editable model. A PTGB file contains
the flattened yarn geometry represented by
`pytexgen.gpu_voxelizer.SnapshotBundle`; it does not contain enough semantic
information to reconstruct an editable `CTextile`.

## Byte Layout

All integer fields and multi-byte array values use little-endian byte order.

| Offset | Size | Value |
| ---: | ---: | --- |
| 0 | 8 | Magic bytes `50 54 47 42 0d 0a 1a 0a` |
| 8 | 2 | Unsigned format major version, currently `1` |
| 10 | 2 | Unsigned format minor version, currently `0` |
| 12 | 8 | Unsigned UTF-8 JSON header length |
| 20 | variable | JSON header |
| aligned | variable | Raw array payloads |

The fixed prefix is the little-endian structure `<8sHHQ`. The first payload
byte is the next 64-byte-aligned offset after the JSON header. Every array
payload also begins at a 64-byte-aligned offset relative to that data start.
There is no compression and no pickle payload.

## JSON Header

The header has these required keys:

```json
{
  "format": "pytexgen.prepared_geometry",
  "format_version": 1,
  "alignment": 64,
  "arrays": {
    "positions": {
      "dtype": "<f4",
      "shape": [200, 3],
      "offset": 0,
      "nbytes": 2400,
      "sha256": "..."
    }
  },
  "metadata": {}
}
```

`offset` is relative to the aligned data start. `sha256` is computed over the
exact raw payload bytes. Readers must reject missing arrays, overlapping or
unaligned offsets, object dtypes, inconsistent byte counts, and payloads that
extend beyond the file.

## Required Arrays

The array directory contains exactly these entries, in this order:

| Name | Shape | Meaning |
| --- | --- | --- |
| `positions` | `(N, 3)` | Slave-node positions |
| `tangents` | `(N, 3)` | Unit yarn tangents |
| `ups` | `(N, 3)` | Unit yarn up vectors |
| `sides` | `(N, 3)` | Tangent-cross-up side vectors |
| `node_offsets` | `(Y + 1,)` | Per-yarn ranges in node arrays |
| `sections` | `(S, 2)` | Flattened section polygon points |
| `section_offsets` | `(Y + 1,)` | Per-yarn ranges in `sections` |
| `translations` | `(T, 3)` | Periodic-image translations |
| `translation_offsets` | `(Y + 1,)` | Per-yarn translation ranges |
| `aabb` | `(2, 3)` | Domain lower and upper corners |

Offset arrays are monotonic integer arrays, start at zero, and end at the
length of their associated flat array. Geometry arrays are numeric C-order
arrays. Writers preserve float precision while canonicalizing byte order.

## Compatibility Rules

- Readers of v1 accept only prefix version `1.0` and header
  `format_version: 1`.
- A format change that alters required arrays or their interpretation requires
  a new major version.
- New informational keys may be added to `metadata` without changing the
  format version.
- Header validation is sufficient for fast trusted-cache startup. Use checksum
  validation when receiving a cache from another machine or storage system.

The reference implementation is
`pytexgen.batch.save_prepared_geometry(...)` and
`pytexgen.batch.load_prepared_geometry(...)`.
