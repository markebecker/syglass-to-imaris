"""
Determine the true per-axis block stride by cross-checking children against their PARENT.

Every previous attempt compared slices to other slices and asked whether they "look alike".
That fails on real data: the masks are large smooth blobs, so any two nearby slices agree
0.8-1.0 whether or not one is a copy of the other.  Three different similarity metrics all
foundered on this.

This test has no such weakness, because it uses an INDEPENDENT encoding of the same data.
In the octree, a block at level L-1 covers exactly the region its eight level-L children
cover, at half the resolution.  So:

    reconstruct a parent's region from its children, placing them at a candidate stride
    downsample the result 2x
    compare against what the parent block actually stores

Only the correct stride reproduces the parent.  A stride one too large shifts every second
child, and the mismatch shows up immediately.  Smoothness cannot rescue a wrong answer
here, because the parent is a fixed target that was written independently.

Comparison is done on 1-D profiles through the volume rather than whole blocks, which keeps
it fast enough for pure Python.

Usage:
    python probe_parent.py path/to/file.syk
    python probe_parent.py path/to/file.syk --span 8 --lines 400

Standard library only.  Read-only.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_syk import (HEADER_BYTES, BLOCK_HEADER_BYTES, FVGU,
                         read_u16, block_level, block_position)


def line(payload, cx, cy, cz, axis, a, b):
    """
    1-D profile along `axis` at transverse coordinates (a, b).
    Flat index is z*cy*cx + y*cx + x; axis 2 = X, 1 = Y, 0 = Z.
    """
    if axis == 2:                       # vary x; a=y, b=z
        s = b * cy * cx + a * cx
        return payload[s:s + cx]
    if axis == 1:                       # vary y; a=x, b=z
        s = b * cy * cx + a
        return payload[s:s + cy * cx:cx]
    s = b * cx + a                      # vary z; a=x, b=y
    return payload[s::cy * cx]


def child_of(parent_id: int, dx: int, dy: int, dz: int) -> int:
    """Block id of a child, given the x=LSB / z=MSB ordering."""
    return parent_id * 8 + 1 + (dx | (dy << 1) | (dz << 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("syk")
    ap.add_argument("--span", type=int, default=8,
                    help="how many strides below the block dim to test (default 8)")
    ap.add_argument("--lines", type=int, default=300,
                    help="profiles to compare per candidate (default 300)")
    ap.add_argument("--parents", type=int, default=6,
                    help="parent blocks to use per axis (default 6)")
    ap.add_argument("--scan-parents", type=int, default=40,
                    help="parent blocks to examine while hunting for painted ones")
    args = ap.parse_args()

    size = os.path.getsize(args.syk)
    with open(args.syk, "rb") as f:
        head = f.read(HEADER_BYTES)
        _, _, cx, cy, cz = struct.unpack_from("<5I", head, 0)
        payload = cx * cy * cz * 2
        rec = BLOCK_HEADER_BYTES + payload
        n_est = (size - HEADER_BYTES) // rec

        offsets = {}
        off = HEADER_BYTES
        for _ in range(n_est):
            f.seek(off)
            hdr = f.read(BLOCK_HEADER_BYTES)
            if len(hdr) < BLOCK_HEADER_BYTES or hdr[0:4] != FVGU:
                break
            pl, bid, _lod, _flg = struct.unpack("<QIII", hdr[4:])
            if pl == payload:
                offsets[bid] = off
            off += rec

        max_level = max(block_level(b) for b in offsets)
        print(f"block dims {cx} x {cy} x {cz}; depth {max_level}\n")

        cache = {}
        def read_block(bid):
            if bid not in cache:
                # Must hold every block a whole axis needs (parent + 2 children, times
                # --parents), or each candidate stride re-reads them all from disk.
                if len(cache) > 64:
                    cache.clear()
                f.seek(offsets[bid] + BLOCK_HEADER_BYTES)
                cache[bid] = read_u16(f.read(payload))
            return cache[bid]

        for axis, name, dim in ((2, "X", cx), (1, "Y", cy), (0, "Z", cz)):
            # Parents whose two children along this axis both exist AND that carry paint —
            # an empty parent yields no evidence at any stride.
            step_xyz = (1 if axis == 2 else 0,      # axis 2 = X, 1 = Y, 0 = Z
                        1 if axis == 1 else 0,
                        1 if axis == 0 else 0)
            candidates = []
            for pid in sorted(offsets):
                if block_level(pid) != max_level - 1:
                    continue
                lo = child_of(pid, 0, 0, 0)
                hi = child_of(pid, *step_xyz)
                if lo in offsets and hi in offsets:
                    candidates.append((pid, lo, hi))
            usable = []
            for pid, lo, hi in candidates[:args.scan_parents]:
                painted = sum(1 for v in read_block(pid) if v)
                if painted:
                    usable.append((painted, pid, lo, hi))
            usable.sort(key=lambda t: -t[0])
            usable = [(pid, lo, hi) for _n, pid, lo, hi in usable[:args.parents]]

            print(f"=== {name} (block dim {dim}) — {len(usable)} parent(s) usable ===")
            if not usable:
                print("    no parent with both children present; cannot measure\n")
                continue

            results = []
            for s in range(dim - args.span, dim + 1):
                match = total = 0
                for pid, lo_id, hi_id in usable:
                    P = read_block(pid)
                    A = read_block(lo_id)
                    B = read_block(hi_id)
                    done = 0
                    # transverse samples inside the lower child (so no transverse
                    # child boundary complicates the mapping)
                    step = max(1, (s // 2) // 12)
                    for j in range(0, min(s // 2, dim // 2), step):
                        for k in range(0, min(s // 2, dim // 2), step):
                            if done >= args.lines:
                                break
                            pl_ = line(P, cx, cy, cz, axis, j, k)
                            # eight level-L voxels collapse into one parent voxel; take
                            # the max, the usual choice for a label pyramid
                            a0 = line(A, cx, cy, cz, axis, 2 * j, 2 * k)
                            a1 = line(A, cx, cy, cz, axis, 2 * j + 1, 2 * k)
                            a2 = line(A, cx, cy, cz, axis, 2 * j, 2 * k + 1)
                            a3 = line(A, cx, cy, cz, axis, 2 * j + 1, 2 * k + 1)
                            b0 = line(B, cx, cy, cz, axis, 2 * j, 2 * k)
                            b1 = line(B, cx, cy, cz, axis, 2 * j + 1, 2 * k)
                            b2 = line(B, cx, cy, cz, axis, 2 * j, 2 * k + 1)
                            b3 = line(B, cx, cy, cz, axis, 2 * j + 1, 2 * k + 1)

                            def recon(x):
                                if x < s:
                                    return max(a0[x], a1[x], a2[x], a3[x])
                                x -= s
                                if x >= dim:
                                    return 0
                                return max(b0[x], b1[x], b2[x], b3[x])

                            # The parent's unique content [0, s) spans level-L voxels
                            # [0, 2s) — i.e. ALL of child A's unique region and all of
                            # child B's.  Covering only [0, s//2) would never consult B
                            # and so could not test the stride at all.
                            for i in range(s):
                                pv = pl_[i]
                                cv = max(recon(2 * i), recon(2 * i + 1))
                                if pv or cv:
                                    total += 1
                                    if pv == cv:
                                        match += 1
                            done += 1
                        if done >= args.lines:
                            break
                r = (match / total) if total else float("nan")
                results.append((s, r, total))

            # A small stride compares fewer voxels, so it can score 1.0 on a handful of
            # them by luck.  Only trust candidates backed by a decent share of the
            # largest sample; report the rest but do not let them win.
            max_t = max((t for _s, _r, t in results), default=0)
            floor_t = max(50, int(0.2 * max_t))
            eligible = [(s, r, t) for s, r, t in results if t >= floor_t and r == r]
            best = max(eligible, key=lambda t: (t[1], t[2])) if eligible else None

            print(f"    {'stride':>7s} {'parent match':>13s} {'voxels':>10s}")
            for s, r, t in results:
                mark = ""
                if best and s == best[0]:
                    mark = "   <== best"
                elif t < floor_t:
                    mark = "   (too few voxels to trust)"
                if s == (dim - 3 if name == "X" else dim - 1):
                    mark += "  [reader uses this]"
                rs = "   n/a" if r != r else f"{r:13.4f}"
                print(f"    {s:7d} {rs:>13s} {t:10d}{mark}")
            if best:
                print(f"    => stride {best[0]} reproduces the parent best "
                      f"({best[1]:.4f} over {best[2]} voxels)\n")
            else:
                print("    => not enough evidence on this axis\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
