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


def test_field_extents(chk) -> None:
    """
    AddSurface spaces samples over (size - 1) intervals with samples ON the extent
    borders, so the extents handed to Imaris must be inset by half a voxel; the invariant
    is that Imaris's resulting sample spacing equals the true voxel size.
    """
    ds_lo = np.zeros(3)
    per = np.ones(3)
    offset, crop = np.array([10, 20, 30]), np.array([5, 7, 9])
    lo, hi = XT._field_extents(ds_lo, per, offset, crop)
    chk("field extents start half a voxel inside the crop edge",
        np.allclose(lo, offset + 0.5), f"(got {lo})")
    chk("field extents give Imaris a sample spacing of one voxel",
        np.allclose((hi - lo) / (crop - 1), per))

    per = np.array([0.5, 2.0, 3.25])
    ds_lo = np.array([-100.0, 40.0, 7.5])
    lo, hi = XT._field_extents(ds_lo, per, offset, crop)
    chk("field extents: anisotropic spacing equals per-voxel exactly",
        np.allclose((hi - lo) / (crop - 1), per))
    chk("field extents sit inside the crop's outer edges",
        np.all(lo > ds_lo + offset * per) and
        np.all(hi < ds_lo + (offset + crop) * per))


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


def test_components(chk) -> None:
    """
    6-connected component labeling: the scipy path and the NumPy run-length union-find
    fallback must be interchangeable, so both are pinned to the same answers.
    """
    def parts(comp, n):
        """Partition as a set of coordinate frozensets — label numbering is arbitrary."""
        return {frozenset(zip(*np.nonzero(comp == cid))) for cid in range(1, n + 1)}

    two = np.zeros((10, 10, 10), bool)
    two[1:3, 1:3, 1:3] = True
    two[6:9, 6:9, 6:9] = True

    diag = np.zeros((4, 4, 4), bool)     # touching only diagonally = NOT connected
    diag[1, 1, 1] = True
    diag[2, 2, 1] = True

    # U-shape: two arms whose runs meet only at the far row — the classic case where two
    # provisional labels must be merged late by the union-find.
    u = np.zeros((5, 1, 4), bool)
    u[:, 0, 0] = True
    u[:, 0, 3] = True
    u[4, 0, :] = True

    single = np.zeros((3, 3, 3), bool)
    single[1, 1, 1] = True

    empty = np.zeros((3, 3, 3), bool)

    for name, vol, want in [("two blobs", two, 2), ("diagonal pair", diag, 2),
                            ("U-shape", u, 1), ("single voxel", single, 1),
                            ("empty", empty, 0)]:
        cf, nf = XT._label_components_numpy(vol)
        cd, nd = XT._label_components(vol)
        chk(f"components fallback: {name} -> {want}", nf == want, f"(got {nf})")
        chk(f"components dispatch: {name} matches fallback",
            nd == nf and parts(cd, nd) == parts(cf, nf))
        chk(f"components {name}: labels cover exactly the mask",
            np.array_equal(cf > 0, vol))

    try:
        from scipy.ndimage import label as nd_label
    except Exception:
        nd_label = None
    if nd_label is not None:
        blobby = rng.random((24, 24, 24)) > 0.75     # many speckle components
        cs, ns = nd_label(blobby)
        cf, nf = XT._label_components_numpy(blobby)
        chk("components fallback matches scipy on speckle",
            ns == nf and parts(cs, ns) == parts(cf, nf), f"(scipy {ns}, numpy {nf})")
    else:
        print("skip  components scipy parity (scipy not installed)")


def test_prep_label(chk) -> None:
    """
    The signed fields Imaris meshes, one per disconnected component: uint8 read as int8,
    so 100 is inside (+100) and 200 is outside (-56).  Anything non-positive would
    produce no zero crossing and no surface.
    """
    vol = np.zeros((60, 40, 40), np.uint16)
    vol[10:20, 12:22, 14:24] = 3     # 1000 voxels
    vol[30:35, 5:10, 5:10] = 3       # 125 voxels
    vol[50, 30, 30] = 3              # 1-voxel speck
    vol[2, 2, 2] = 5                 # another label, must not leak in

    prep = XT._prep_label(vol, 3, sigma=0.0)
    comps = prep["components"]
    chk("prep finds every disconnected component",
        prep["n_components"] == 3 and len(comps) == 3)
    chk("prep orders components largest first",
        [c[2] for c in comps] == [1000, 125, 1])
    chk("prep component voxels total the label",
        sum(c[2] for c in comps) == prep["n_painted"] == 1126)

    field, off, _vox, _kept = comps[0]
    chk("prep writes 100/200", set(np.unique(field)) == {100, 200})
    chk("prep marks exactly the painted voxels as inside",
        (field.view(np.int8) > 0).sum() == 1000)
    chk("prep offset places the blob correctly",
        field[tuple(np.array([15, 17, 19]) - np.array(off))] == 100)

    inside = set()
    for f, o, _v, _k in comps:
        inside |= {tuple(np.array(p) + o) for p in zip(*np.nonzero(f.view(np.int8) > 0))}
    chk("prep components cover the label exactly (in global coordinates)",
        inside == set(zip(*np.nonzero(vol == 3))))
    chk("prep returns None for an absent label", XT._prep_label(vol, 99) is None)

    smoothed = XT._prep_label(vol, 3, sigma=1.5)
    off_s = smoothed["components"][0][1]
    chk("prep margin grows with sigma",
        (np.array(off) - np.array(off_s) > 0).all()
        and smoothed["components"][0][0].shape > field.shape)
    s = smoothed["components"][0][0].view(np.int8)
    chk("smoothed field is signed either side of the boundary",
        (s > 0).any() and (s < 0).any())
    chk("prep skips components the blur erases (1-voxel speck at sigma 1.5)",
        len(smoothed["components"]) == 2 and smoothed["n_vanished"] == 1
        and smoothed["vanished_voxels"] == 1)

    # Isolation: two blobs 3 voxels apart, blurred.  Each component's field must stay on
    # its own side of the gap (no neighbour tail leaking positive voxels into the crop)
    # and fall back to "outside" before its crop border on every face.
    close = np.zeros((30, 20, 20), np.uint16)
    close[5:10, 5:15, 5:15] = 7
    close[13:18, 5:15, 5:15] = 7
    sp = XT._prep_label(close, 7, sigma=1.5)
    chk("prep splits blobs closer than the blur reach", len(sp["components"]) == 2)
    positives, sides, borders_ok = [], [], True
    for f, o, _v, _k in sp["components"]:
        s = f.view(np.int8)
        borders_ok &= bool((s[0] <= 0).all() and (s[-1] <= 0).all()
                           and (s[:, 0] <= 0).all() and (s[:, -1] <= 0).all()
                           and (s[:, :, 0] <= 0).all() and (s[:, :, -1] <= 0).all())
        P = {tuple(np.array(p) + o) for p in zip(*np.nonzero(s > 0))}
        positives.append(P)
        xs = {p[0] for p in P}
        sides.append("lo" if max(xs) < 12 else ("hi" if min(xs) > 11 else "mixed"))
    chk("prep smoothing keeps components isolated",
        set(sides) == {"lo", "hi"} and not (positives[0] & positives[1]))
    chk("prep smoothed fields are outside at every crop border", borders_ok)


def main() -> int:
    chk = Checker("pipeline (NumPy paths)")
    test_upload(chk)
    test_inventory(chk)
    test_extents(chk)
    test_field_extents(chk)
    test_smoothing(chk)
    test_octree_maths(chk)
    test_components(chk)
    test_prep_label(chk)
    return chk.done()


if __name__ == "__main__":
    raise SystemExit(main())
