"""
Tests for the pure-NumPy pipeline between reading the mask and handing it to Imaris:
upload chunking, label inventory, extent mapping, smoothing, octree index maths and
per-label field construction.

None of this needs Imaris.  The COM calls themselves cannot be tested off the workstation,
but the byte layout handed to them can be, and is.

Run:  python tests/test_pipeline.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from syk_fixtures import Checker, load_xtension

XT = load_xtension()
rng = np.random.default_rng(0)


class Recorder:
    """Stands in for both an IDataSet and the progress bar."""

    def __init__(self):
        self.calls, self.ticks = [], 0

    def SetDataSubVolumeAs1DArrayBytes(self, *a):
        self.calls.append(a)

    def band(self):
        self.ticks += 1


def test_upload(chk) -> None:
    """
    The upload must stay under the Ice message limit, tick the progress bar exactly as
    many times as _upload_ticks promised, and reassemble into the original field.
    A drift between ticks and calls would silently desync the progress bar.
    """
    for shape, budget in [((7, 5, 4), 900_000), ((7, 5, 4), 40), ((7, 5, 4), 12),
                          ((13, 11, 9), 100), ((3, 3, 3), 1)]:
        field = rng.integers(0, 255, size=shape).astype(np.uint8)
        r = Recorder()
        XT._upload_field(r, field, r, budget)
        declared = XT._upload_ticks(*shape, budget)

        # Bands are never split along X, so one row is the floor; (3,3,3) at budget 1
        # sits below that floor deliberately, to pin the behaviour there.
        worst = max(len(c[0]) for c in r.calls)
        limit = max(budget, shape[0])
        chk(f"upload {shape} budget {budget}: every call within budget", worst <= limit,
            f"(largest {worst}, limit {limit})")
        chk(f"upload {shape} budget {budget}: ticks match _upload_ticks",
            declared == r.ticks, f"({declared} vs {r.ticks})")

        # Imaris wants X fastest, then Y, then Z; rebuild from the blobs and compare.
        rebuilt = np.zeros(shape, np.uint8)
        for blob, _c, y0, z0, _a, _b, nx, ny, nz in r.calls:
            chunk = np.frombuffer(blob, np.uint8).reshape(nz, ny, nx).transpose(2, 1, 0)
            rebuilt[:, y0:y0 + ny, z0:z0 + nz] = chunk
        chk(f"upload {shape} budget {budget}: bytes reassemble exactly",
            np.array_equal(rebuilt, field))

    # the module constant must be read at call time, not frozen as a default argument
    field = rng.integers(0, 255, size=(13, 11, 9)).astype(np.uint8)
    explicit = Recorder(); XT._upload_field(explicit, field, explicit, 100)
    saved = XT._ICE_CHUNK_BYTES
    XT._ICE_CHUNK_BYTES = 100
    try:
        via_global = Recorder(); XT._upload_field(via_global, field, via_global)
    finally:
        XT._ICE_CHUNK_BYTES = saved
    chk("upload honours _ICE_CHUNK_BYTES at call time",
        explicit.calls == via_global.calls)


def test_inventory(chk) -> None:
    """Slab-wise histogram must equal a single bincount, and drop implausible IDs."""
    vol = rng.integers(0, 6, size=(200, 30, 20)).astype(np.uint16)
    vol[0, 0, :3] = [18024, 30825, 150]
    ids, counts = XT._inventory_labels(vol)
    chk("inventory equals a single bincount",
        np.array_equal(counts, np.bincount(vol.ravel(), minlength=2 ** 16)))
    chk("inventory drops implausible IDs", ids == [1, 2, 3, 4, 5], f"(got {ids})")
    chk("inventory covers every voxel", int(counts.sum()) == vol.size)


def test_extents(chk) -> None:
    """
    The mask is scaled onto the .ims EXTENTS using the full syGlass grid, so the grid size
    participates in placement.  The syGlass grid deliberately need not match the .ims voxel
    grid — assuming it must sent an earlier investigation badly astray.
    """
    class Geom:
        ext_min = np.array([-1800.75, -11707.2, 0.0])
        ext_max = np.array([8505.91, 10970.3, 6020.0])
        size = np.array([2167, 4768, 1204])
        voxel_size = (ext_max - ext_min) / size

    g = Geom()
    clip_info = (304, 2864, 48, 2128, 4784, 1200)     # real mbls005 numbers
    shape = (1824, 1920, 1152)
    lo, hi, _full = XT._clip_extents(g, clip_info, shape)
    per = (g.ext_max - g.ext_min) / np.array(clip_info[3:])
    chk("extents scale by the full syGlass grid",
        np.allclose(lo, g.ext_min + np.array(clip_info[:3]) * per)
        and np.allclose(hi, g.ext_min + (np.array(clip_info[:3]) + np.array(shape)) * per))

    lo2, _, _ = XT._clip_extents(g, (304, 2864, 48, 9999, 9999, 9999), shape)
    chk("extents depend on the grid size (stretch-to-fit)", not np.allclose(lo, lo2))

    lo3, hi3, _ = XT._clip_extents(g, None, shape)
    chk("unclipped extents equal the dataset extents",
        np.allclose(lo3, g.ext_min) and np.allclose(hi3, g.ext_max))


def test_smoothing(chk) -> None:
    """
    The fallback smoother must not erode the array edge, and must put the 0.5 crossing —
    where Imaris meshes the surface — exactly on the voxel boundary.
    """
    solid = XT._smooth_mask_3d(np.ones((5, 5, 5), bool), iterations=3)
    chk("smoothing preserves a solid block (no edge darkening)", np.allclose(solid, 1.0),
        f"(min {solid.min():.4f})")

    half = np.zeros((10, 5, 5), bool); half[:5] = True
    sm = XT._smooth_mask_3d(half, iterations=2)
    inside, outside = sm[4, 2, 2], sm[5, 2, 2]
    chk("smoothing puts the 0.5 crossing on the voxel boundary",
        inside > 0.5 > outside and abs((inside + outside) - 1.0) < 1e-5,
        f"(inside {inside:.4f}, outside {outside:.4f})")
    chk("smoothing stays within [0, 1]", sm.min() >= 0.0 and sm.max() <= 1.0)
    chk("smoothing is monotonic across the edge",
        all(sm[i, 2, 2] >= sm[i + 1, 2, 2] for i in range(9)))


def test_octree_maths(chk) -> None:
    """Breadth-first block ids map to octree level and grid position."""
    chk("block level from id",
        [XT._syk_block_level(i) for i in [0, 1, 8, 9, 72, 73]] == [0, 1, 1, 2, 2, 3])
    chk("root block position", XT._syk_block_position(0) == (0, 0, 0, 0))
    chk("child order is x=LSB, z=MSB",
        [XT._syk_block_position(i)[1:] for i in range(1, 9)] ==
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
         (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)])
    chk("rgba packs to signed int32",
        XT._pack_rgba(255, 0, 0, 255) == int(np.uint32(0xFF0000FF).astype(np.int32)))


def test_prep_label(chk) -> None:
    """
    The signed field Imaris meshes: uint8 read as int8, so 100 is inside (+100) and 200 is
    outside (-56).  Anything non-positive would produce no zero crossing and no surface.
    """
    vol = np.zeros((40, 40, 40), np.uint16)
    vol[10:20, 12:22, 14:24] = 3
    vol[30, 30, 30] = 5

    field, off, n_painted, _n_kept = XT._prep_label(vol, 3, sigma=0.0)
    chk("prep crops to one label only", n_painted == 1000, f"(painted {n_painted})")
    chk("prep writes 100/200", set(np.unique(field)) == {100, 200})
    chk("prep marks exactly the painted voxels as inside",
        (field.view(np.int8) > 0).sum() == 1000)
    chk("prep offset places the blob correctly",
        field[tuple(np.array([15, 17, 19]) - np.array(off))] == 100)
    chk("prep returns None for an absent label", XT._prep_label(vol, 99) is None)

    smoothed, off_s, _, _ = XT._prep_label(vol, 3, sigma=2.0)
    chk("prep margin grows with sigma",
        (np.array(off) - np.array(off_s) >= 0).all() and smoothed.shape > field.shape)
    chk("smoothed field is signed either side of the boundary",
        (smoothed.view(np.int8) > 0).any() and (smoothed.view(np.int8) < 0).any())


def main() -> int:
    chk = Checker("pipeline (NumPy paths)")
    test_upload(chk)
    test_inventory(chk)
    test_extents(chk)
    test_smoothing(chk)
    test_octree_maths(chk)
    test_prep_label(chk)
    return chk.done()


if __name__ == "__main__":
    raise SystemExit(main())
