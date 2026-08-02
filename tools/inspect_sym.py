"""
Read-only scan of a syGlass .sym project file for volume metadata.

A .sym is a ZIP-wrapped LevelDB.  Properly parsing LevelDB needs its block and
varint format, but we do not need to: the keys are stored as plain ASCII, so a
string scan reveals the schema, and a numeric scan finds any field holding the
volume dimensions.

This exists to answer one question the .syk cannot: does syGlass store the
volume at the .ims dimensions, or does it round up to a grid that is convenient
for the octree — and if so, is the .ims volume placed at the origin of that grid
or centred within it?

Usage:
    python inspect_sym.py path/to/file.sym --ims-dims 2160 2560 1281

Standard library only.  Nothing is written and the archive is only read.
"""

from __future__ import annotations

import argparse
import re
import struct
import zipfile
from collections import Counter

PRINTABLE = re.compile(rb"[ -~]{4,}")


def scan_strings(blob: bytes, limit: int = 400):
    """Distinct printable runs, longest first — LevelDB keys show up here."""
    seen = Counter()
    for m in PRINTABLE.finditer(blob):
        seen[m.group().decode("ascii", "replace")] += 1
    return seen.most_common(limit)


def scan_numbers(blob: bytes, targets, tol: int = 64):
    """
    Every offset where a uint32 / uint64 / float64 is close to a target value.

    Reports the neighbouring bytes too, because a volume dimension is almost
    always stored next to its siblings — finding X with Y and Z beside it is far
    stronger evidence than any single hit.
    """
    hits = []
    n = len(blob)
    for off in range(0, n - 8):
        u32 = struct.unpack_from("<I", blob, off)[0]
        for name, t in targets:
            if abs(u32 - t) <= tol and u32 != 0:
                ctx = struct.unpack_from("<6I", blob, off) if off + 24 <= n else ()
                hits.append((off, "u32", u32, name, t, ctx))
                break
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sym")
    ap.add_argument("--ims-dims", nargs=3, type=int, metavar=("X", "Y", "Z"), required=True)
    ap.add_argument("--max-hits", type=int, default=60)
    args = ap.parse_args()
    targets = list(zip("XYZ", args.ims_dims))

    with zipfile.ZipFile(args.sym) as zf:
        members = zf.namelist()
        print("=" * 78)
        print(f"ARCHIVE: {len(members)} member(s)")
        print("=" * 78)
        for info in zf.infolist():
            print(f"  {info.filename:40s} {info.file_size:12,d} bytes")

        blob = b"".join(zf.read(m) for m in members)

    print(f"\n  total uncompressed = {len(blob):,} bytes")

    print("\n" + "=" * 78)
    print("KEY-LIKE STRINGS (LevelDB keys are plain ASCII)")
    print("=" * 78)
    strings = scan_strings(blob)
    keyish = [(s, c) for s, c in strings if "::" in s]
    other = [(s, c) for s, c in strings if "::" not in s and len(s) >= 6]
    if keyish:
        print("  containing '::' (syGlass key namespace):")
        for s, c in keyish[:60]:
            print(f"    x{c:<5d} {s[:100]}")
    else:
        print("  (no '::' keys found)")
    print("\n  other printable runs (first 40):")
    for s, c in other[:40]:
        print(f"    x{c:<5d} {s[:100]}")

    print("\n" + "=" * 78)
    print(f"NUMERIC SCAN for values near .ims dims {args.ims_dims}")
    print("=" * 78)
    print("  Looking for the volume dimensions, and for a rounded-up grid size")
    print("  (which would confirm syGlass pads the volume for the octree).\n")
    hits = scan_numbers(blob, targets)
    if not hits:
        print("  no uint32 within 64 of any .ims dimension found")
    for off, kind, val, axname, tgt, ctx in hits[:args.max_hits]:
        delta = val - tgt
        print(f"  offset {off:10,d}  {kind} = {val:<8d} (.ims {axname}={tgt}, "
              f"{delta:+d})   neighbours: {ctx}")
    if len(hits) > args.max_hits:
        print(f"  ... and {len(hits) - args.max_hits} more")

    print("\n  INTERPRETATION")
    print("  - three hits close together, each equal to an .ims dimension")
    print("      => syGlass stores the true volume size; the octree grid is padding")
    print("  - three hits equal to 16*blockdim instead")
    print("      => syGlass stores the PADDED grid, and the .ims sits inside it")
    print("  - a further triple of small numbers beside them")
    print("      => likely the offset of the volume within the grid (centring)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
