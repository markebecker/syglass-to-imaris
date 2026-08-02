"""
Test whether .syk blocks carry PADDING slices — slices that duplicate another slice of the
SAME block rather than data from a neighbour.

Motivation: the reported artifact is a single slice appearing at a regular interval where
it does not belong.  If a block contains a copy of one of its own slices, placing that copy
emits the duplicate a block-width away from the original — exactly that symptom.

This has never been tested.  Earlier tools only compared block A against its NEIGHBOUR B,
which cannot see self-duplication at all, and the one self-check that existed searched from
j=0 and so trivially matched slice 0 against itself.

Method: for well-painted blocks, compare each slice near either end against EVERY other
slice of the same block, and report the best match for each along with the block's own
baseline agreement between unrelated slices.  Nothing is assumed about which slice might be
duplicated or which one it copies — the duplicate need not involve slice 0, the pad may be
at the start rather than the end, it may be any width, or there may be none.  Read the
'agree' column against the baseline: a duplicate stands far above it.

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

        SAMPLE = 1500          # voxels per slice comparison — plenty for a ratio
        for axis, name, dim in ((2, "X", cx), (1, "Y", cy), (0, "Z", cz)):
            print(f"=== {name} (block dim {dim}) ===")
            # Cache every slice of every sampled block, subsampled, so the all-pairs
            # comparison below is cheap.
            cache = []
            for _n, _bid, blk in picked:
                sl = []
                for j in range(dim):
                    full = take_slice(blk, cx, cy, cz, axis, j)
                    step = max(1, len(full) // SAMPLE)
                    sl.append(full[::step])
                cache.append(sl)

            def pair(j, k):
                m = t = 0
                for sl in cache:
                    for a, b in zip(sl[j], sl[k]):
                        if a or b:
                            t += 1
                            if a == b:
                                m += 1
                return (m / t) if t else float("nan"), t

            # baseline: typical agreement between two unrelated interior slices
            bl = []
            for a, b in ((dim // 3, 2 * dim // 3), (dim // 4, 3 * dim // 4),
                         (dim // 2, dim // 2 + 1 if dim > 2 else 0)):
                r, t = pair(a, b)
                if t:
                    bl.append(r)
            base = sum(bl) / len(bl) if bl else float("nan")
            print(f"    baseline agreement, two unrelated slices: {base:.3f}")
            print(f"    {'slice j':>8s} {'best k':>7s} {'agree':>7s} {'exact?':>7s}"
                  f"   {'2nd best':>9s}   note")

            edges = list(range(0, args.edge)) + list(range(dim - args.edge, dim))
            flagged = []
            for j in edges:
                scores = []
                for k in range(dim):
                    if k == j:
                        continue
                    r, t = pair(j, k)
                    if t:
                        scores.append((r, k))
                if not scores:
                    continue
                scores.sort(reverse=True)
                r1, k1 = scores[0]
                r2, k2 = scores[1] if len(scores) > 1 else (float("nan"), -1)
                # confirm the winner on the FULL slice, not the subsample
                exact = all(take_slice(blk, cx, cy, cz, axis, j)
                            == take_slice(blk, cx, cy, cz, axis, k1)
                            for _n, _bid, blk in picked)
                note = ""
                if r1 > base + 0.25 or exact:
                    note = "  <== duplicates another slice"
                    flagged.append((j, k1, r1, exact))
                where = (f"[{dim - j} from END]" if j >= dim - args.edge
                         else f"[{j} from START]")
                print(f"    {j:8d} {k1:7d} {r1:7.3f} {str(exact):>7s}   "
                      f"{r2:9.3f}   {where}{note}")

            if not flagged:
                print("    => no slice duplicates any other slice: NO padding on this axis")
            else:
                pads = sorted({j for j, _k, _r, _e in flagged})
                print(f"    => duplicated slice indices: {pads}")
                trailing = [j for j in pads if j >= dim - args.edge]
                if trailing:
                    keep = min(trailing)
                    print(f"    => trailing padding starts at {keep}; unique content is "
                          f"[0, {keep}) so the stride is {keep}")
                leading = [j for j in pads if j < args.edge]
                if leading:
                    print(f"    => LEADING duplicates at {leading} — the block may begin "
                          f"with pad, which would shift placement, not just widen it")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
