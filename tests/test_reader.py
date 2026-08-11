"""
End-to-end tests for the .syk reader, against synthetic files with known contents.

These cover the three properties that were hardest to get right, each of which was wrong
at some point and produced visible artifacts:

  * a UNIFORM octree, where every leaf sits at the deepest level
  * an ADAPTIVE octree, where a block with no stored children is a leaf at its own coarser
    resolution and must be upsampled into place — missing this dropped 478 of 512 regions
    on a real file
  * the per-axis block trim, i.e. how much of each block is overlap with its neighbour

Run:  python tests/test_reader.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from syk_fixtures import (Checker, block_from, downsample2, load_xtension, write_syk)

XT = load_xtension()
SENTINEL_A, SENTINEL_B = 18024, 30825


def _read(blocks, dims, **kw):
    """Write a synthetic .syk, parse it, and clean up."""
    tmp = tempfile.NamedTemporaryFile(suffix=".syk", delete=False)
    tmp.close()
    try:
        write_syk(tmp.name, dims, blocks)
        return XT._read_mask_from_syk(tmp.name, **kw)
    finally:
        os.unlink(tmp.name)


def test_uniform(chk) -> None:
    """Every leaf at the deepest level, plus sentinel handling."""
    CX, CY, CZ = 8, 6, 5
    trim = (3, 1, 1)
    ax, ay, az = CX - trim[0], CY - trim[1], CZ - trim[2]
    full = (2 * ax, 2 * ay, 2 * az)

    truth = np.zeros(full, np.uint16)
    truth[2:8, 2:8, 1:7] = 1                 # blob straddling every block boundary
    truth[4, 4, 3] = SENTINEL_A              # sentinel buried inside the blob
    truth[9, 9, 7] = SENTINEL_B              # sentinel alone in the background

    blocks = {0: np.ones((CX, CY, CZ), np.uint16)}   # root: nonzero so bbox = full grid
    for bid in range(1, 9):
        _lv, ix, iy, iz = XT._syk_block_position(bid)
        # fill=SENTINEL_A puts junk in the trailing slices the reader must discard
        blocks[bid] = block_from(truth, (ix * ax, iy * ay, iz * az), (CX, CY, CZ),
                                 fill=SENTINEL_A)

    vol, clip = _read(blocks, (CX, CY, CZ), trim=trim)

    expected = truth.copy()
    expected[4, 4, 3] = 0      # sentinels erased; no seam fill, so the hole remains
    expected[9, 9, 7] = 0

    chk("uniform: parser returned a volume", vol is not None)
    if vol is None:
        return
    chk("uniform: clip_info reports the full grid", clip == (0, 0, 0, *full), f"(got {clip})")
    chk("uniform: shape matches the full grid", vol.shape == full, f"(got {vol.shape})")
    chk("uniform: reconstruction matches ground truth exactly",
        np.array_equal(vol, expected), f"(differing voxels: {int((vol != expected).sum())})")
    chk("uniform: no sentinel value survives",
        not np.isin(vol, [SENTINEL_A, SENTINEL_B]).any())
    chk("uniform: isolated sentinel became background, not a phantom label",
        vol[9, 9, 7] == 0, f"(got {vol[9, 9, 7]})")

    ids, _counts = XT._inventory_labels(vol)
    chk("uniform: inventory sees only the real label", ids == [1], f"(got {ids})")

    # Erasing sentinels must not change the meshed surface: a sentinel reads as
    # "not this label" exactly as a zero does.
    field, off, n_painted, _ = XT._prep_label(vol, 1, sigma=0.0)
    with_sent = vol.copy(); with_sent[4, 4, 3] = SENTINEL_A
    field2, off2, _, _ = XT._prep_label(with_sent, 1, sigma=0.0)
    chk("uniform: erasing sentinels is surface-neutral",
        off == off2 and np.array_equal(field, field2))
    chk("uniform: painted voxel count matches the blob",
        n_painted == int((expected == 1).sum()),
        f"({n_painted} vs {int((expected == 1).sum())})")


def test_adaptive(chk) -> None:
    """
    A file where one region is subdivided to the deepest level and the rest is stored
    coarse.  The coarse leaves must be upsampled into place, not discarded.
    """
    CX, CY, CZ = 8, 6, 5
    trim = (3, 1, 1)
    ax, ay, az = CX - trim[0], CY - trim[1], CZ - trim[2]
    fine = (4 * ax, 4 * ay, 4 * az)          # depth 2 -> 4 leaf slots per axis

    truth = np.zeros(fine, np.uint16)
    truth[0:2 * ax, 0:2 * ay, 0:2 * az] = 1          # inside level-1 block (0,0,0): COARSE
    truth[2 * ax:4 * ax, 2 * ay:4 * ay, 2 * az:4 * az] = 2   # level-1 (1,1,1): SUBDIVIDED

    level = {2: truth, 1: downsample2(truth), 0: downsample2(downsample2(truth))}

    def make(bid):
        lv, ix, iy, iz = XT._syk_block_position(bid)
        return block_from(level[lv], (ix * ax, iy * ay, iz * az), (CX, CY, CZ),
                          fill=SENTINEL_A)

    # Store the root and all eight level-1 blocks, but level-2 children for only the
    # (1,1,1) octant.  The other seven level-1 blocks are therefore childless leaves.
    subdivided = next(b for b in range(1, 9) if XT._syk_block_position(b)[1:] == (1, 1, 1))
    ids = [0] + list(range(1, 9)) + [subdivided * 8 + 1 + i for i in range(8)]
    blocks = {bid: make(bid) for bid in ids}

    vol, _clip = _read(blocks, (CX, CY, CZ), trim=trim)

    chk("adaptive: parser returned a volume", vol is not None)
    if vol is None:
        return
    chk("adaptive: volume is at the FINEST resolution", vol.shape == fine,
        f"(got {vol.shape}, want {fine})")
    n1, n2 = int((vol == 1).sum()), int((vol == 2).sum())
    chk("adaptive: coarse-stored region was NOT discarded", n1 > 0,
        f"({n1} voxels; reading only the deepest level drops these)")
    chk("adaptive: subdivided region still present", n2 > 0, f"({n2} voxels)")
    chk("adaptive: no sentinel leaked through", not np.isin(vol, [SENTINEL_A]).any())
    chk("adaptive: reconstruction matches ground truth exactly", np.array_equal(vol, truth),
        f"(differing voxels: {int((vol != truth).sum())})")


def test_trim(chk) -> None:
    """The trim is honoured per axis, including the no-trim case."""
    CX, CY, CZ = 8, 6, 5
    for trim in [(3, 1, 1), (0, 0, 0), (2, 1, 0)]:
        ax, ay, az = CX - trim[0], CY - trim[1], CZ - trim[2]
        full = (2 * ax, 2 * ay, 2 * az)
        truth = np.zeros(full, np.uint16)
        truth[1:, 1:, 1:] = 1
        blocks = {0: np.ones((CX, CY, CZ), np.uint16)}
        for bid in range(1, 9):
            _lv, ix, iy, iz = XT._syk_block_position(bid)
            blocks[bid] = block_from(truth, (ix * ax, iy * ay, iz * az), (CX, CY, CZ),
                                     fill=777)
        vol, _clip = _read(blocks, (CX, CY, CZ), trim=trim)
        ok = vol is not None and vol.shape == full and np.array_equal(vol, truth)
        chk(f"trim {trim}: reconstructs exactly", ok,
            f"(shape {None if vol is None else vol.shape}, want {full})")


def test_locked(chk) -> None:
    """
    A PermissionError on open — how a Windows sharing violation from syGlass surfaces —
    must raise _SykLockedError, not be reported as an empty file.
    """
    import builtins
    tmp = tempfile.NamedTemporaryFile(suffix=".syk", delete=False)
    tmp.close()
    try:
        write_syk(tmp.name, (8, 6, 5), {0: np.ones((8, 6, 5), np.uint16)})
        real_open = builtins.open
        target = tmp.name

        def deny(path, *args, **kwargs):
            if path == target:
                raise PermissionError(13, "sharing violation", path)
            return real_open(path, *args, **kwargs)

        builtins.open = deny
        try:
            chk("locked: probe reports locked", XT._syk_is_locked(target))
            raised = False
            try:
                XT._read_mask_from_syk(target)
            except XT._SykLockedError:
                raised = True
            chk("locked: reader raises _SykLockedError", raised)
        finally:
            builtins.open = real_open

        chk("locked: probe reports readable once released", not XT._syk_is_locked(target))
        vol, _clip = XT._read_mask_from_syk(target)
        chk("locked: read succeeds once released", vol is not None)
    finally:
        os.unlink(tmp.name)


def main() -> int:
    chk = Checker("reader (synthetic .syk end-to-end)")
    test_uniform(chk)
    test_adaptive(chk)
    test_trim(chk)
    test_locked(chk)
    return chk.done()


if __name__ == "__main__":
    raise SystemExit(main())
