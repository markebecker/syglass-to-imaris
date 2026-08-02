"""
Test whether .syk blocks carry PADDING slices — slices that duplicate another slice of the
SAME block rather than data from a neighbour.

This is the hypothesis behind the reader's `cx - 3` X stride: that a block's trailing
columns are copies of its own column 0.  If true, placing those columns emits a duplicate
of column 0 at the block's high-X end — one block-width away from where it belongs, which
is the "a single X slice appears displaced at regular intervals" artifact.

It has never actually been tested.  Earlier tools only compared block A against its
NEIGHBOUR B, which cannot see self-duplication, and the one self-check that existed
searched j from 0 and so trivially matched slice 0 against itself.

Method: for well-painted blocks, compare every slice j against every other slice k of the
same block and report pairs that agree far above the block's own baseline.  No assumption
about which slices might be padding, so it will also catch a leading pad, a differently
sized pad, or none at all.

Usage:
    python probe_padding.py path/to/file.syk
    python probe_padding.py path/to/file.syk --blocks 6 --edge 6

Standard library only.  Read-only.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_syk import (HEADER_BYTES, BLOCK_HEADER_BYTES, FVGU,
                         read_u16, block_level, block_position, take_slice)


def agree(sa, sb):
    """(matching, considered, exact) over voxels where either slice is painted."""
    match = total = 0
    for a, b in zip(sa, sb):
        if a or b:
            total += 1
            if a == b:
                match += 1
    return match, total, (sa == sb)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("syk")
    ap.add_argument("--blocks", type=int, default=6, help="painted blocks to sample")
    ap.add_argument("--scan", type=int, default=120, help="blocks to look through")
    ap.add_argument("--edge", type=int, default=6,
                    help="how many slices at each end to test against the rest")
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
        leaves = [b for b in offsets if block_level(b) == max_level]
        print(f"block dims {cx} x {cy} x {cz}; depth {max_level}; {len(leaves)} leaves\n")

        # pick the most-painted blocks we can find within the scan budget
        picked = []
        for bid in leaves[:args.scan]:
            f.seek(offsets[bid] + BLOCK_HEADER_BYTES)
            blk = read_u16(f.read(payload))
            n = sum(1 for v in blk if v)
            if n:
                picked.append((n, bid, blk))
        picked.sort(key=lambda t: -t[0])
        picked = picked[:args.blocks]
        if not picked:
            print("no painted blocks found")
            return 1
        print(f"using {len(picked)} block(s); painted voxels: "
              f"{', '.join(str(n) for n, _, _ in picked)}\n")

        for axis, name, dim in ((2, "X", cx), (1, "Y", cy), (0, "Z", cz)):
            print(f"=== {name} (block dim {dim}) ===")
            # baseline: how much do two arbitrary interior slices agree?
            base_m = base_t = 0
            for _n, _bid, blk in picked:
                a = take_slice(blk, cx, cy, cz, axis, dim // 3)
                b = take_slice(blk, cx, cy, cz, axis, 2 * dim // 3)
                m, t, _ = agree(a, b)
                base_m += m
                base_t += t
            base = (base_m / base_t) if base_t else float("nan")
            print(f"    baseline agreement between two interior slices: {base:.3f}")

            # every edge slice against every other slice of the same block
            found = []
            edges = list(range(0, args.edge)) + list(range(dim - args.edge, dim))
            for j in edges:
                for k in range(dim):
                    if k == j:
                        continue
                    tot_m = tot_t = 0
                    exact = 0
                    for _n, _bid, blk in picked:
                        sj = take_slice(blk, cx, cy, cz, axis, j)
                        sk = take_slice(blk, cx, cy, cz, axis, k)
                        m, t, e = agree(sj, sk)
                        tot_m += m
                        tot_t += t
                        exact += 1 if e else 0
                    if tot_t == 0:
                        continue
                    r = tot_m / tot_t
                    if r > max(0.95, base + 0.25):
                        found.append((r, exact, j, k, tot_t))
            if not found:
                print("    no slice duplicates another slice of the same block "
                      "-> NO padding on this axis")
            else:
                found.sort(reverse=True)
                print(f"    {'slice j':>8s} {'== slice k':>11s} {'agree':>7s} "
                      f"{'exact':>7s} {'voxels':>9s}")
                for r, exact, j, k, t in found[:12]:
                    where = ""
                    if j >= dim - args.edge:
                        where = f"  (j is {dim - j} from the END)"
                    elif j < args.edge:
                        where = f"  (j is {j} from the START)"
                    print(f"    {j:8d} {k:11d} {r:7.3f} {exact:4d}/{len(picked):<3d} "
                          f"{t:9d}{where}")
                pads = sorted({j for _r, _e, j, _k, _t in found})
                print(f"    => duplicated slice indices: {pads}")
                trailing = [j for j in pads if j >= dim - args.edge]
                if trailing:
                    keep = min(trailing)
                    print(f"    => trailing padding starts at {keep}, so the unique "
                          f"content is [0, {keep}) and the stride is {keep}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
