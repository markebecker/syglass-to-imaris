"""
Shared helpers for the XTension tests: import the XTension outside Imaris, and build
synthetic .syk files with known contents.

The XTension imports ImarisLib at module scope, which only exists inside Imaris, so it is
stubbed here.  Everything the tests exercise is pure NumPy and file parsing — the COM paths
cannot be tested off the workstation.
"""

from __future__ import annotations

import os
import struct
import sys
import types

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEADER_BYTES = 36
BLOCK_HEADER_BYTES = 24
FVGU = b"fvgu"


def load_xtension():
    """Import XT_import_from_syglass with ImarisLib stubbed out."""
    if "ImarisLib" not in sys.modules:
        stub = types.ModuleType("ImarisLib")
        stub.ImarisLib = object
        sys.modules["ImarisLib"] = stub
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    import XT_import_from_syglass
    return XT_import_from_syglass


class Checker:
    """Minimal pass/fail reporter so the tests need no third-party runner."""

    def __init__(self, title: str) -> None:
        self.failures: list[str] = []
        print(f"\n=== {title} ===")

    def __call__(self, name: str, cond, extra: str = "") -> None:
        print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}")
        if not cond:
            self.failures.append(name)

    def done(self) -> int:
        print(f"  -> {'all passed' if not self.failures else self.failures}")
        return 1 if self.failures else 0


def write_syk(path: str, block_dims, blocks: dict) -> None:
    """
    Write a synthetic .syk.

    `block_dims` is (cx, cy, cz); `blocks` maps block id -> an (x, y, z) uint16 array of
    exactly those dimensions.  Blocks are stored (cz, cy, cx) as syGlass does, behind the
    36-byte file header and a 24-byte record header each.

    Which ids are present is what makes a file uniform or adaptive: a block whose children
    are absent is a leaf, and the reader must treat it as one.
    """
    cx, cy, cz = block_dims
    payload = cx * cy * cz * 2
    buf = bytearray(struct.pack("<5I", 0, 0, cx, cy, cz) + b"\x00" * 16)
    for bid in sorted(blocks):
        arr = blocks[bid]
        assert arr.shape == (cx, cy, cz), f"block {bid}: {arr.shape} != {(cx, cy, cz)}"
        buf += FVGU + struct.pack("<QIII", payload, bid, 0, 0)
        buf += arr.transpose(2, 1, 0).astype("<u2").tobytes()
    with open(path, "wb") as f:
        f.write(bytes(buf))


def downsample2(a: np.ndarray) -> np.ndarray:
    """2x max-downsample, the usual reduction for a label pyramid."""
    s = [d // 2 * 2 for d in a.shape]
    b = a[:s[0], :s[1], :s[2]]
    return b.reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2).max(axis=(1, 3, 5))


def block_from(source: np.ndarray, origin, block_dims, fill=0) -> np.ndarray:
    """
    Carve a (cx, cy, cz) block out of `source` starting at `origin`, padding past the edge.

    `fill` is written into the region beyond `source`, which lets a test put recognisable
    junk in the trailing slices the reader is supposed to discard.
    """
    cx, cy, cz = block_dims
    ox, oy, oz = origin
    out = np.full((cx, cy, cz), fill, np.uint16)
    sub = source[ox:ox + cx, oy:oy + cy, oz:oz + cz]
    out[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
    return out
