"""
Report the .syk octree's structure: is it a uniform grid at the deepest level, or an
ADAPTIVE octree where a block with no stored children is itself a leaf?

This distinction decides whether the mask reader is correct in principle.  It currently
composites only blocks at the deepest level present, which is right if the octree is a
uniform grid, and loses data if it is adaptive.

The suspicion comes from the block census.  One test file stores every block at levels 0-3
but only 240 of 4096 at level 4.  If a level-3 block contains mask, at least one of its
eight children must too, so 512 occupied level-3 blocks cannot be covered by 240 level-4
blocks.  Either most of those level-3 blocks are empty placeholders, or they are leaves
holding data at their own resolution — and this tool tells the two apart by simply looking
at which blocks actually contain anything.

For each octree level it reports:
    stored        blocks present in the file
    non-empty     blocks containing at least one nonzero voxel
    childless     blocks with no stored children (leaves, if the octree is adaptive)
    LEAF DATA     non-empty AND childless — data the current reader would discard
                  unless the block sits at the deepest level

Usage:
    python octree_structure.py path/to/file.syk

Standard library only.  Read-only.  Reads every block once, so allow a minute or two.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_syk import HEADER_BYTES, BLOCK_HEADER_BYTES, FVGU, block_level


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("syk")
    args = ap.parse_args()

    size = os.path.getsize(args.syk)
    with open(args.syk, "rb") as f:
        head = f.read(HEADER_BYTES)
        _, _, cx, cy, cz = struct.unpack_from("<5I", head, 0)
        payload = cx * cy * cz * 2
        rec = BLOCK_HEADER_BYTES + payload
        n_est = (size - HEADER_BYTES) // rec
        print(f"block dims {cx} x {cy} x {cz}; {n_est} records\n")

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

        # any(raw) short-circuits at C speed, so "does this block contain anything"
        # costs almost nothing for a painted block and one scan for an empty one
        nonempty = {}
        for i, (bid, boff) in enumerate(sorted(offsets.items())):
            f.seek(boff + BLOCK_HEADER_BYTES)
            nonempty[bid] = any(f.read(payload))
            if i % 100 == 0:
                print(f"    scanning… {i}/{len(offsets)}", end="\r", file=sys.stderr)
        print(" " * 40, end="\r", file=sys.stderr)

    has_child = defaultdict(bool)
    for bid in offsets:
        if bid:
            has_child[(bid - 1) // 8] = True

    max_level = max(block_level(b) for b in offsets)
    per = defaultdict(lambda: [0, 0, 0, 0])   # stored, non-empty, childless, leaf-data
    for bid in offsets:
        lv = block_level(bid)
        row = per[lv]
        row[0] += 1
        if nonempty[bid]:
            row[1] += 1
        if not has_child[bid]:
            row[2] += 1
            if nonempty[bid]:
                row[3] += 1

    print(f"{'level':>6s} {'slots':>7s} {'stored':>8s} {'non-empty':>10s} "
          f"{'childless':>10s} {'LEAF DATA':>10s}")
    lost = 0
    for lv in sorted(per):
        stored, ne, childless, leafdata = per[lv]
        slots = 8 ** lv
        flag = ""
        if lv < max_level and leafdata:
            flag = "   <== READER DISCARDS THIS"
            lost += leafdata
        print(f"{lv:6d} {slots:7d} {stored:8d} {ne:10d} {childless:10d} "
              f"{leafdata:10d}{flag}")

    print()
    print(f"deepest level = {max_level}")
    if lost:
        print(f"ADAPTIVE OCTREE: {lost} non-empty block(s) below the deepest level have no")
        print("children, so they are leaves holding their region's mask at their own")
        print("resolution.  The reader composites only deepest-level blocks and therefore")
        print("drops all of them — regions covered by those blocks come out empty.")
    else:
        print("UNIFORM OCTREE: every non-empty block below the deepest level has stored")
        print("children, so all mask data really does live at the deepest level and")
        print("compositing only that level loses nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
