"""
Read-only forensic dump of a syGlass .syk octree.

Answers questions the XTension currently has to assume:

  1. What is in the 36-byte file header?  (Answer so far: magic, version, the
     three block dimensions, and the octree depth — but NO volume dimensions.)

  2. What is in the ~64 KB footer that ends with the ASCII tag "INDX"?  It is
     4096 * 16 bytes plus a 24-byte trailer, and there are exactly 4096 leaf
     slots in a depth-4 octree, so it is almost certainly a per-leaf index.  It
     may also carry the volume dimensions, which would settle the geometry.

  3. What is the real per-axis block stride?  Rather than assuming an apron
     width, this measures it.  For adjacent blocks A and B it asks which slice
     index j of A reproduces slice 0 of B.

     The decisive test is EXACT identity of the whole slice, background
     included.  A duplicated apron is a byte-for-byte copy, so it scores 1.00.
     A merely *adjacent* slice of a smooth mask also scores high on a
     painted-voxel-only comparison — that is spatial autocorrelation, not
     duplication — which is why both metrics are reported side by side.  Trust
     the "exact" column.

Usage:
    python inspect_syk.py path/to/file.syk --ims-dims 2160 2560 1281

Standard library only — no numpy — so it runs under any Python 3, not just the
interpreter Imaris is configured with.  Nothing is written and the file is only
ever opened for reading, so this is safe to run against live data.  syGlass must
not hold the file open (close the project first) or Windows will refuse the read.
"""

from __future__ import annotations

import argparse
import os
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


def dump_header(raw: bytes, ims) -> None:
    print("=" * 78)
    print("FILE HEADER (36 bytes)")
    print("=" * 78)
    print("  raw hex:", raw.hex(" ", 4))
    u32 = struct.unpack("<9I", raw)
    print("\n  as uint32 (9):")
    known = {2: "cx (block dim X)", 3: "cy (block dim Y)", 4: "cz (block dim Z)",
             0: "magic", 1: "version?", 6: "octree depth?"}
    for i, v in enumerate(u32):
        note = ""
        if ims:
            for ax, d in zip("XYZ", ims):
                if v and abs(int(v) - d) <= 32:
                    note = f"   <== within 32 of .ims {ax}={d}"
        print(f"    [{i}] bytes {i*4:2d}-{i*4+3:2d} = {v:12d}   {known.get(i,''):18s}{note}")


def dump_footer(f, footer_start: int, footer_len: int, n_leaves_possible: int, ims) -> None:
    """Dump the trailing INDX region and try to read it as a per-leaf table."""
    print("\n" + "=" * 78)
    print("FOOTER / INDX REGION")
    print("=" * 78)
    f.seek(footer_start)
    foot = f.read(footer_len)
    print(f"  offset {footer_start:,}  length {footer_len:,} bytes")
    print(f"  last 16 bytes: {foot[-16:].hex(' ', 4)}   tail ASCII: {foot[-4:]!r}")

    body_len = footer_len - 24                    # 24-byte trailer
    if body_len <= 0 or body_len % 4:
        print("  (unexpected footer size; skipping structured interpretation)")
        return
    words = array("I")
    words.frombytes(foot[:body_len - (body_len % 4)])
    if sys.byteorder == "big":
        words.byteswap()

    nonzero = [(i, w) for i, w in enumerate(words) if w]
    print(f"  body = {body_len:,} bytes = {len(words):,} uint32, "
          f"{len(nonzero):,} of them nonzero")
    if body_len % 16 == 0:
        entries = body_len // 16
        print(f"  body divides into {entries} x 16-byte entries "
              f"({'MATCHES' if entries == n_leaves_possible else 'does not match'} "
              f"the {n_leaves_possible} leaf slots)")

    print("\n  distinct nonzero uint32 values (up to 20 most common):")
    for v, c in Counter(w for _, w in nonzero).most_common(20):
        note = ""
        if ims:
            for ax, d in zip("XYZ", ims):
                if abs(int(v) - d) <= 64:
                    note = f"   <== near .ims {ax}={d}"
        print(f"    {v:12d}  (0x{v:08x})  x{c:<6d}{note}")

    print("\n  first 12 nonzero 16-byte entries (as u64,u32,u32 and as 4x u32):")
    shown = 0
    for e in range(body_len // 16):
        chunk = foot[e * 16:(e + 1) * 16]
        if not any(chunk):
            continue
        q, a, b = struct.unpack("<QII", chunk)
        w = struct.unpack("<4I", chunk)
        print(f"    entry {e:5d}: u64={q:<16d} u32={a:<10d} u32={b:<10d} | {w}")
        shown += 1
        if shown >= 12:
            break
    if shown == 0:
        print("    (the entire body is zero)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("syk")
    ap.add_argument("--ims-dims", nargs=3, type=int, metavar=("X", "Y", "Z"),
                    help="voxel dimensions of the matching .ims, for comparison")
    ap.add_argument("--pairs", type=int, default=8,
                    help="adjacent block pairs to USE per axis (default 8)")
    ap.add_argument("--scan", type=int, default=60,
                    help="max pairs to examine while hunting for painted ones (default 60)")
    ap.add_argument("--min-paint", type=int, default=500,
                    help="min painted voxels at a boundary for a pair to count (default 500)")
    args = ap.parse_args()
    ims = tuple(args.ims_dims) if args.ims_dims else None

    size = os.path.getsize(args.syk)
    with open(args.syk, "rb") as f:
        head = f.read(HEADER_BYTES)
        if len(head) < HEADER_BYTES:
            print("file too short")
            return 1
        dump_header(head, ims)

        _, _, cx, cy, cz = struct.unpack_from("<5I", head, 0)
        payload = cx * cy * cz * 2
        rec = BLOCK_HEADER_BYTES + payload
        n_est = (size - HEADER_BYTES) // rec
        leftover = (size - HEADER_BYTES) - n_est * rec

        print("\n" + "=" * 78)
        print("BLOCK TABLE")
        print("=" * 78)
        print(f"  block dims cx,cy,cz = {cx}, {cy}, {cz}")
        print(f"  payload / record    = {payload:,} / {rec:,} bytes")
        print(f"  file size           = {size:,} bytes")
        print(f"  whole records       = {n_est}    trailing bytes = {leftover:,}")

        offsets, levels, flags = {}, Counter(), Counter()
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
                flags[flg] += 1
            off += rec

        max_level = max(levels)
        n_grid = 2 ** max_level
        leaves = {}
        for bid in offsets:
            if block_level(bid) == max_level:
                _lv, ix, iy, iz = block_position(bid)
                leaves[(ix, iy, iz)] = bid
        print(f"  blocks per level    = {dict(sorted(levels.items()))}")
        print(f"  flags histogram     = {dict(flags)}")
        print(f"  deepest level {max_level} -> {n_grid} blocks/axis, "
              f"{len(leaves)} of {n_grid**3} leaves present")

        if leftover > 0:
            dump_footer(f, HEADER_BYTES + n_est * rec, leftover, n_grid ** 3, ims)

        cache = {}
        def read_block(bid):
            if bid not in cache:
                if len(cache) > 6:
                    cache.clear()
                f.seek(offsets[bid] + BLOCK_HEADER_BYTES)
                cache[bid] = read_u16(f.read(payload))
            return cache[bid]

        print("\n" + "=" * 78)
        print("STRIDE MEASUREMENT")
        print("=" * 78)
        print("  Which slice j of block A reproduces slice 0 of neighbour B?")
        print("  exact  = fraction of pairs where the ENTIRE slice is byte-identical")
        print("           (this is the decisive column: an apron is a literal copy)")
        print("  paint  = agreement over painted voxels only — inflated by mask")
        print("           autocorrelation, since adjacent slices simply look alike\n")

        implied = {}
        for arr_axis, name, dim, step in ((2, "X", cx, (1, 0, 0)),
                                          (1, "Y", cy, (0, 1, 0)),
                                          (0, "Z", cz, (0, 0, 1))):
            exact = [0] * dim
            pa, pt = [0] * dim, [0] * dim
            used = examined = 0
            for pos, bid in leaves.items():
                if used >= args.pairs or examined >= args.scan:
                    break
                nb = tuple(p + s for p, s in zip(pos, step))
                if nb not in leaves:
                    continue
                examined += 1
                A, B = read_block(bid), read_block(leaves[nb])
                b0 = take_slice(B, cx, cy, cz, arr_axis, 0)
                # only use pairs with real signal at the boundary
                if sum(1 for v in b0 if v) < args.min_paint:
                    continue
                for j in range(dim):
                    aj = take_slice(A, cx, cy, cz, arr_axis, j)
                    if aj == b0:
                        exact[j] += 1
                    for va, vb in zip(aj, b0):
                        if va or vb:
                            pt[j] += 1
                            if va == vb:
                                pa[j] += 1
                used += 1

            if used == 0:
                print(f"  {name}: no pair with >= {args.min_paint} painted boundary voxels "
                      f"(examined {examined}) — cannot measure")
                implied[name] = None
                continue

            paint = [(pa[j] / pt[j]) if pt[j] else -1.0 for j in range(dim)]
            order = sorted(range(dim), key=lambda j: (exact[j], paint[j]), reverse=True)[:5]
            print(f"  {name} (block dim {dim}): {used} usable pairs of {examined} examined")
            print(f"      {'j':>5s}  {'exact':>12s}  {'paint':>6s}  {'voxels':>9s}")
            for j in order:
                tag = "  <- last slice" if j == dim - 1 else (
                      "  <- code assumes" if j == dim - 3 and name == "X" else "")
                print(f"      {j:5d}  {exact[j]:5d}/{used:<6d}  {paint[j]:6.2f}  "
                      f"{pt[j]:9d}{tag}")
            best = order[0]
            if exact[best] == used:
                implied[name] = best
                print(f"      => EXACT duplicate at j={best}: stride = {best} "
                      f"(apron = {dim - best} voxel(s))")
            elif exact[best]:
                implied[name] = best
                print(f"      => partial duplicate at j={best} ({exact[best]}/{used} pairs); "
                      f"stride probably {best}, but not consistent")
            else:
                implied[name] = dim
                print(f"      => NO exact duplicate on any j -> blocks ABUT, stride = {dim}")

        print("\n" + "=" * 78)
        print("IMPLIED GEOMETRY")
        print("=" * 78)
        cur = {"X": cx - 3, "Y": cy - 1, "Z": cz - 1}
        for i, name in enumerate("XYZ"):
            s = implied.get(name)
            if s is None:
                print(f"  {name}: not measured")
                continue
            gn, gc = n_grid * s, n_grid * cur[name]
            line = (f"  {name}: measured stride {s:4d} -> grid {gn:5d}   |   "
                    f"code assumes {cur[name]:4d} -> grid {gc:5d}")
            if ims:
                line += (f"   |   .ims {ims[i]:5d}: measured "
                         f"{'covers' if gn >= ims[i] else 'TOO SMALL'}, assumed "
                         f"{'covers' if gc >= ims[i] else 'TOO SMALL'}")
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
