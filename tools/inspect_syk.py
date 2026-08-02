"""
Read-only forensic dump of a syGlass .syk octree.

Answers two questions the XTension currently has to assume:

  1. What is in the 36-byte file header?  The mask reader uses only fields 2-4
     (the block dimensions) and ignores field 0, field 1, and the trailing 16
     bytes.  If syGlass records the true volume dimensions anywhere, they are in
     there, and knowing them removes the guesswork from the extent mapping.

  2. What is the real per-axis block stride?  Rather than assuming an apron
     width, this measures it: for two adjacent blocks A and B, it finds which
     slice index j of A duplicates slice 0 of B.  That j IS the stride, because
     A's unique region is [0, stride) and B begins at stride.  If no j matches,
     the blocks abut and the stride is the full block dimension.

Usage:
    python inspect_syk.py path/to/file.syk --ims-dims 2160 2560 1281

Standard library only — no numpy — so it runs under any Python 3, not just the
interpreter Imaris is configured with.  Nothing is written and the file is only
ever opened for reading, so this is safe to run against live data.  syGlass must
not hold the file open (close the project first) or Windows will refuse the read.
"""

from __future__ import annotations

import argparse
import struct
import sys
from array import array
from collections import Counter

HEADER_BYTES = 36
BLOCK_HEADER_BYTES = 24
FVGU = b"fvgu"


def read_u16(data: bytes) -> array:
    """Little-endian uint16 view of a block payload."""
    a = array("H")
    a.frombytes(data)
    if sys.byteorder == "big":
        a.byteswap()
    return a


def take_slice(payload: array, cx: int, cy: int, cz: int, axis: int, j: int):
    """
    One slice of a (cz, cy, cx) C-ordered payload, perpendicular to `axis`,
    where axis 2 = X, 1 = Y, 0 = Z.  Flat index is z*cy*cx + y*cx + x.
    """
    if axis == 2:                       # X: every cx-th element starting at j
        return payload[j::cx]
    if axis == 0:                       # Z: one contiguous plane
        return payload[j * cy * cx:(j + 1) * cy * cx]
    out = array("H")                    # Y: cz contiguous runs of cx
    for z in range(cz):
        base = z * cy * cx + j * cx
        out.extend(payload[base:base + cx])
    return out


def agreement(sa, sb):
    """(matching, considered) over voxels where either slice is painted."""
    match = total = 0
    for a, b in zip(sa, sb):
        if a or b:
            total += 1
            if a == b:
                match += 1
    return match, total


def block_level(block_id: int) -> int:
    """Octree depth of a block given its breadth-first slot index."""
    if block_id == 0:
        return 0
    level, total, count = 0, 0, 1
    while total + count <= block_id:
        total += count
        count *= 8
        level += 1
    return level


def block_position(block_id: int):
    """Return (level, ix, iy, iz); child order x=LSB, z=MSB."""
    if block_id == 0:
        return 0, 0, 0, 0
    parent, child = (block_id - 1) // 8, (block_id - 1) % 8
    p_lv, p_ix, p_iy, p_iz = block_position(parent)
    return (p_lv + 1,
            2 * p_ix + (child & 1),
            2 * p_iy + ((child >> 1) & 1),
            2 * p_iz + ((child >> 2) & 1))


def dump_header(raw: bytes, ims_dims) -> None:
    print("=" * 78)
    print("FILE HEADER (36 bytes)")
    print("=" * 78)
    print("  raw hex:", raw.hex(" ", 4))

    u32 = struct.unpack("<9I", raw)
    u16 = struct.unpack("<18H", raw)
    f32 = struct.unpack("<9f", raw)
    u64 = struct.unpack("<4Q", raw[:32])

    print("\n  as uint32 (9):")
    for i, v in enumerate(u32):
        role = {2: "cx (block dim X)", 3: "cy (block dim Y)", 4: "cz (block dim Z)"}.get(i, "")
        note = ""
        if ims_dims:
            for ax, d in zip("XYZ", ims_dims):
                if v and abs(int(v) - d) <= 32:
                    note = f"   <== within 32 of .ims {ax}={d}"
        print(f"    [{i}] bytes {i*4:2d}-{i*4+3:2d} = {v:12d}   {role}{note}")

    print("\n  as float32 (9):", "  ".join(f"{v:.4g}" for v in f32))
    print("  as uint64 (4): ", "  ".join(str(v) for v in u64))
    print("  as uint16 (18):", " ".join(str(v) for v in u16))
    if ims_dims:
        print(f"\n  .ims dims for comparison: {ims_dims[0]} x {ims_dims[1]} x {ims_dims[2]}")
        print("  Look above for any field matching these, or a multiple/half of them.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("syk")
    ap.add_argument("--ims-dims", nargs=3, type=int, metavar=("X", "Y", "Z"),
                    help="voxel dimensions of the matching .ims, for comparison")
    ap.add_argument("--pairs", type=int, default=12,
                    help="max adjacent block pairs to sample per axis (default 12)")
    args = ap.parse_args()
    ims = tuple(args.ims_dims) if args.ims_dims else None

    import os
    size = os.path.getsize(args.syk)
    with open(args.syk, "rb") as f:
        head = f.read(HEADER_BYTES)
        if len(head) < HEADER_BYTES:
            print("file too short"); return 1
        dump_header(head, ims)

        _, _, cx, cy, cz = struct.unpack_from("<5I", head, 0)
        payload = cx * cy * cz * 2
        stride_bytes = BLOCK_HEADER_BYTES + payload
        n_est = (size - HEADER_BYTES) // stride_bytes
        leftover = (size - HEADER_BYTES) - n_est * stride_bytes

        print("\n" + "=" * 78)
        print("BLOCK TABLE")
        print("=" * 78)
        print(f"  block dims cx,cy,cz     = {cx}, {cy}, {cz}")
        print(f"  payload / record stride = {payload:,} / {stride_bytes:,} bytes")
        print(f"  file size               = {size:,} bytes")
        print(f"  whole records after hdr = {n_est}")
        print(f"  LEFTOVER TRAILING BYTES = {leftover:,}"
              f"{'   <== possible footer/metadata, inspect it' if leftover else '   (none)'}")

        offsets, levels, lods, flags = {}, Counter(), Counter(), Counter()
        off = HEADER_BYTES
        for _ in range(n_est):
            f.seek(off)
            hdr = f.read(BLOCK_HEADER_BYTES)
            if len(hdr) < BLOCK_HEADER_BYTES or hdr[0:4] != FVGU:
                break
            pl, bid, lod, flg = struct.unpack("<QIII", hdr[4:])
            if pl == payload:
                offsets[bid] = off
                levels[block_level(bid)] += 1
                lods[lod] += 1
                flags[flg] += 1
            off += stride_bytes

        if leftover:
            f.seek(size - min(leftover, 256))
            print(f"  trailing bytes (last {min(leftover,256)}): {f.read(min(leftover,256)).hex(' ', 4)}")

        print(f"\n  valid blocks            = {len(offsets)}")
        print(f"  blocks per octree level = {dict(sorted(levels.items()))}")
        print(f"  distinct 'lod' values   = {dict(lods)}")
        print(f"  distinct 'flags' values = {dict(flags)}")

        max_level = max(levels)
        n_grid = 2 ** max_level
        print(f"  deepest level           = {max_level}  ->  {n_grid} blocks per axis")

        leaves = {}
        for bid in offsets:
            if block_level(bid) == max_level:
                _lv, ix, iy, iz = block_position(bid)
                leaves[(ix, iy, iz)] = bid
        print(f"  leaf blocks present     = {len(leaves)} of {n_grid ** 3}")

        cache = {}
        def read_block(bid):
            if bid not in cache:
                if len(cache) > 8:
                    cache.clear()
                f.seek(offsets[bid] + BLOCK_HEADER_BYTES)
                cache[bid] = read_u16(f.read(payload))
            return cache[bid]

        # ---------------------------------------------------------------
        # Measure the true stride per axis.
        # ---------------------------------------------------------------
        print("\n" + "=" * 78)
        print("STRIDE MEASUREMENT")
        print("=" * 78)
        print("  For adjacent blocks A,B: which slice j of A equals slice 0 of B?")
        print("  That j is the stride (A owns [0,j), B starts at j).")
        print("  'abut' = no j matches, so the stride is the full block dimension.\n")

        implied = {}
        # payload is (cz, cy, cx): axis 2 = X, 1 = Y, 0 = Z
        for arr_axis, name, dim, step in ((2, "X", cx, (1, 0, 0)),
                                          (1, "Y", cy, (0, 1, 0)),
                                          (0, "Z", cz, (0, 0, 1))):
            agree = [0] * dim; total = [0] * dim
            self_agree = [0] * dim; self_total = [0] * dim
            pairs = 0
            for pos, bid in leaves.items():
                if pairs >= args.pairs:
                    break
                nb = tuple(p + s for p, s in zip(pos, step))
                if nb not in leaves:
                    continue
                A, B = read_block(bid), read_block(leaves[nb])
                b0 = take_slice(B, cx, cy, cz, arr_axis, 0)
                a0 = take_slice(A, cx, cy, cz, arr_axis, 0)
                for j in range(dim):
                    aj = take_slice(A, cx, cy, cz, arr_axis, j)
                    m, t = agreement(aj, b0)
                    agree[j] += m; total[j] += t
                    m, t = agreement(aj, a0)
                    self_agree[j] += m; self_total[j] += t
                pairs += 1

            if pairs == 0:
                print(f"  {name}: no adjacent painted block pair found — cannot measure")
                implied[name] = None
                continue

            ratio = [(agree[j] / total[j]) if total[j] else -1.0 for j in range(dim)]
            selfr = [(self_agree[j] / self_total[j]) if self_total[j] else -1.0
                     for j in range(dim)]
            order = sorted(range(dim), key=lambda j: ratio[j], reverse=True)[:4]
            print(f"  {name} (block dim {dim}, {pairs} pairs):")
            for j in order:
                if total[j] == 0:
                    continue
                tag = ""
                if j == dim - 1: tag = "  <- last slice (1-voxel apron)"
                elif j == dim - 3: tag = "  <- what the code assumes for X"
                print(f"      j={j:4d}  A[j]==B[0] in {ratio[j]:5.2f} of "
                      f"{total[j]:8d} painted voxels{tag}")
            best = order[0]
            if ratio[best] > 0.80:
                implied[name] = best
                print(f"      => stride = {best}  (= block dim {dim} minus "
                      f"{dim - best} apron voxel(s))")
            else:
                implied[name] = dim
                print(f"      => no slice matches; blocks ABUT, stride = {dim}")
            sbest = max(range(dim), key=lambda j: selfr[j])
            print(f"      (padding check: A[{sbest}] best matches A[0] at "
                  f"{max(0.0, selfr[sbest]):.2f})")

        # ---------------------------------------------------------------
        print("\n" + "=" * 78)
        print("IMPLIED GEOMETRY")
        print("=" * 78)
        cur = {"X": cx - 3, "Y": cy - 1, "Z": cz - 1}
        for i, name in enumerate("XYZ"):
            s = implied.get(name)
            if s is None:
                print(f"  {name}: not measured")
                continue
            grid_new, grid_cur = n_grid * s, n_grid * cur[name]
            line = (f"  {name}: measured stride {s:4d} -> grid {grid_new:5d}   |   "
                    f"code assumes {cur[name]:4d} -> grid {grid_cur:5d}")
            if ims:
                cover_new = "covers" if grid_new >= ims[i] else "TOO SMALL"
                cover_cur = "covers" if grid_cur >= ims[i] else "TOO SMALL"
                line += (f"   |   .ims {ims[i]:5d}: measured {cover_new}, "
                         f"assumed {cover_cur}")
            print(line)
        if ims:
            print("\n  A correct stride must produce a grid >= the .ims dimension:")
            print("  the octree cannot store less data than the volume it came from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
