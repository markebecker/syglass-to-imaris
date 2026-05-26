"""
Imaris XTension: Import syGlass Masks and Counting Points into Imaris.

Runs inside a live Imaris session (Windows only).  Imaris calls
XTImportFromSyGlass(imarisFile) where imarisFile is the numeric handle
for the ImarisLib COM connection.

Usage
-----
1. Open the .ims file in Imaris.
2. Ensure the matching .syk (and optionally .sym) are in the same directory.
3. Run Extensions → Import from syGlass.

Strategy
--------
Masks:
  1. Scan all .syk block headers to inventory the octree.
  2. Read root block (block 0) to estimate the mask bounding box; allocate
     a clipped volume covering only that region plus a small margin.
  3. Composite all blocks coarsest-first into the clipped volume (finer blocks
     overwrite coarser), upsampling each block to fine-grid resolution.
     Ghost slice on X is dropped; Y and Z have none.
  4. Per label: create a uint8 IDataSet (100 = inside, 200 = outside) with
     physical extents matching the .ims dataset, upload slice-by-slice, and
     call ISurfaces.AddSurface — Imaris finds the zero-crossing in the field
     (uint8 is interpreted as signed int8, so 100 = +100, 200 = -56).

Spots (counting points):
  Read countingPoints from .sym LevelDB, convert coordinates back to Imaris
  physical space, create ISpots.  (Fully untested slop.)

Python 3.11 compatibility required (Imaris bundled interpreter).
"""

from __future__ import annotations

import os
import shutil
import struct
import sys
import tempfile
import traceback
import zipfile

import numpy as np

# ImarisLib from Imaris's Python interpreter
import ImarisLib  # type: ignore  # noqa: E402  (Windows COM wrapper)

# ------------------------------------------
# XTension entry point (Imaris calls this)
# -----------------------------------------------------------------------

#   <CustomTools>
#       <Menu>
#           <Item name="Import from syGlass" icon="Python3"
#                 tooltip="Import syGlass masks and counting points into the current scene">
#               <Command>Python3XT::XTImportFromSyGlass(%i)</Command>
#           </Item>
#       </Menu>
#   </CustomTools>

def XTImportFromSyGlass(imarisFile: int) -> None:
    """Main XTension entry point."""
    try:
        _run(imarisFile)
    except Exception:
        tb = traceback.format_exc()
        # Try a couple paths
        for log_path in [
            os.path.join(os.path.expanduser("~"), "pulse_xt_crash.log"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "pulse_xt_crash.log"),
        ]:
            try:
                with open(log_path, "a") as f:
                    f.write(tb)
                print(f"[syglass to imaris] CRASH — traceback written to: {log_path}")
                break
            except Exception:
                pass
        print(tb)  # also try stdout in case console is still open


def _run(imarisFile: int) -> None:
    """Inner implementation (wrapped for crash logging)."""
    vLib = ImarisLib.ImarisLib()
    vApp = vLib.GetApplication(imarisFile)

    # ----------------------------------------------------------------
    # 1. Locate companion syGlass files
    # ----------------------------------------------------------------
    ims_path = vApp.GetCurrentFileName()
    if not ims_path:
        _log_warning(vApp, "No file is currently open in Imaris.")
        return
    # TODO: make this work good and ideally open a window in Imaris

    ims_dir  = os.path.dirname(ims_path)
    ims_stem = os.path.splitext(os.path.basename(ims_path))[0]

    syk_path = os.path.join(ims_dir, ims_stem + ".syk")
    sym_path = os.path.join(ims_dir, ims_stem + ".sym")
    # TODO: what if someone has named the syglass files differently? is this even possible?

    if not os.path.exists(syk_path):
        # Try a case-insensitive search
        for fname in os.listdir(ims_dir):
            if fname.lower() == (ims_stem + ".syk").lower():
                syk_path = os.path.join(ims_dir, fname)
                break
        else:
            _log_warning(vApp, "No .syk file found alongside the open .ims.\n"
                         f"Expected: {syk_path}")
            return

    # ----------------------------------------------------------------
    # 2. Read image dimensions and physical extents from Imaris
    # ----------------------------------------------------------------
    ds = vApp.GetDataSet()
    size_x = ds.GetSizeX()
    size_y = ds.GetSizeY()
    size_z = ds.GetSizeZ()
    ext_min_x = ds.GetExtendMinX()
    ext_min_y = ds.GetExtendMinY()
    ext_min_z = ds.GetExtendMinZ()
    ext_max_x = ds.GetExtendMaxX()
    ext_max_y = ds.GetExtendMaxY()
    ext_max_z = ds.GetExtendMaxZ()

    voxel_size = np.array([
        (ext_max_x - ext_min_x) / size_x,
        (ext_max_y - ext_min_y) / size_y,
        (ext_max_z - ext_min_z) / size_z,
    ])
    print(f"[syglass to imaris] Importing syglass mask[s]...")
    print(f"[syglass to imaris] .ims dims:    {size_x} x {size_y} x {size_z} voxels")
    print(f"[syglass to imaris] .ims extents: X=[{ext_min_x:.3f}, {ext_max_x:.3f}] "
          f"Y=[{ext_min_y:.3f}, {ext_max_y:.3f}] Z=[{ext_min_z:.3f}, {ext_max_z:.3f}]")
    print(f"[syglass to imaris] .ims voxel size (um): {voxel_size[0]:.4f} x {voxel_size[1]:.4f} x {voxel_size[2]:.4f}")

    # ----------------------------------------------------------------
    # 3. Read syGlass label mask
    # ----------------------------------------------------------------
    label_vol = None  # will be (NX, NY, NZ) uint32, 0=background, 1..N=label

    clip_info = None   # (vx0,vy0,vz0, full_nx,full_ny,full_nz) or None

    label_vol, clip_info = _read_mask_from_syk(syk_path, size_x, size_y, size_z)

    if label_vol is None or not np.any(label_vol):
        _log_warning(vApp, "No mask data found in .syk file (or file is empty).")
        return

    all_ids = [int(v) for v in np.unique(label_vol) if v != 0]
    label_ids = [v for v in all_ids if v <= 1000]
    ignored   = [v for v in all_ids if v > 1000]
    # TODO: is it possible anyone might ever make more than 1000 labels?
    # TODO: why do the 18xxx and 31xxx labels even exist?
    print(f"[syglass to imaris] label_vol shape: {label_vol.shape}, dtype: {label_vol.dtype}")
    print(f"[syglass to imaris] Non-zero voxels: {int(np.count_nonzero(label_vol))} / {label_vol.size}")
    for v in all_ids:
        count = int(np.sum(label_vol == v))
        print(f"[syglass to imaris]   label {v}: {count} voxels")
    if ignored:
        print(f"[syglass to imaris] Ignoring {len(ignored)} implausible label IDs (>1000): {ignored}")
        label_vol = label_vol.copy()
        for v in ignored:
            label_vol[label_vol == v] = 0
    print(f"[syglass to imaris] Found {len(label_ids)} labels: {label_ids}")

    # Mask bounding box in physical µm — helps verify coordinate mapping
    nz = np.where(label_vol > 0)
    if len(nz[0]) > 0:
        _nx, _ny, _nz = label_vol.shape
        _vx = (ext_max_x - ext_min_x) / _nx
        _vy = (ext_max_y - ext_min_y) / _ny
        _vz = (ext_max_z - ext_min_z) / _nz
        bx0 = ext_min_x + nz[0].min() * _vx;  bx1 = ext_min_x + nz[0].max() * _vx
        by0 = ext_min_y + nz[1].min() * _vy;  by1 = ext_min_y + nz[1].max() * _vy
        bz0 = ext_min_z + nz[2].min() * _vz;  bz1 = ext_min_z + nz[2].max() * _vz
        print(f"[syglass to imaris] Mask bbox (um): X=[{bx0:.1f}, {bx1:.1f}] ({bx1-bx0:.1f} um wide)")
        print(f"[syglass to imaris]                 Y=[{by0:.1f}, {by1:.1f}] ({by1-by0:.1f} um tall)")
        print(f"[syglass to imaris]                 Z=[{bz0:.1f}, {bz1:.1f}] ({bz1-bz0:.1f} um deep)")

    # ----------------------------------------------------------------
    # 4. Build surfaces via AddSurface with a full-resolution uint8 IDataSet.
    #    AddSurface interprets the dataset as a signed field and finds the
    #    surface at the zero crossing.  uint8 is treated as signed int8:
    #      100 → +100 (inside), 200 → -56 (outside).
    #    The IDataSet is sized to the full-res clipped volume with physical
    #   extents matching the full .ims dataset.  Data is uploaded one z-slice at
    #    a time as tBytes2D = List[bytes] indexed [X][Y] (Ice size limit).
    # ----------------------------------------------------------------
    factory = vApp.GetFactory()

    # Resolve eTypeUInt8 — the enum lives in ImarisLib or a submodule
    eTypeUInt8 = None
    # Try 1: ImarisLib module attribute (in bundled interpreter)
    if hasattr(ImarisLib, 'tType'):
        eTypeUInt8 = ImarisLib.tType.eTypeUInt8
    if eTypeUInt8 is None:
        # Try 2: scan loaded modules for one that exposes tType
        for _mod in list(sys.modules.values()):
            if _mod is not None and hasattr(_mod, 'tType'):
                eTypeUInt8 = _mod.tType.eTypeUInt8
                break
    if eTypeUInt8 is None:
        # Try 3: use integer value directly (eTypeUInt8 = 0 in all Imaris versions I"ve seen)
        print("[syglass to imaris] WARNING: could not resolve tType enum; using raw integer 0 for eTypeUInt8")
        eTypeUInt8 = 0

    vol_nx, vol_ny, vol_nz = label_vol.shape

    # Compute IDataSet physical extents.
    # If the reconstruction was clipped to a sub-region, adjust accordingly.
    if clip_info is not None:
        vx0, vy0, vz0, full_nx, full_ny, full_nz = clip_info
        _vx = (ext_max_x - ext_min_x) / full_nx
        _vy = (ext_max_y - ext_min_y) / full_ny
        _vz = (ext_max_z - ext_min_z) / full_nz
        ds_ext_min_x = ext_min_x + vx0 * _vx
        ds_ext_max_x = ext_min_x + (vx0 + vol_nx) * _vx
        ds_ext_min_y = ext_min_y + vy0 * _vy
        ds_ext_max_y = ext_min_y + (vy0 + vol_ny) * _vy
        ds_ext_min_z = ext_min_z + vz0 * _vz
        ds_ext_max_z = ext_min_z + (vz0 + vol_nz) * _vz
        print(f"[syglass to imaris] IDataSet (clipped): {vol_nx}×{vol_ny}×{vol_nz} voxels  "
              f"full grid {full_nx}×{full_ny}×{full_nz}")
    else:
        ds_ext_min_x, ds_ext_max_x = ext_min_x, ext_max_x
        ds_ext_min_y, ds_ext_max_y = ext_min_y, ext_max_y
        ds_ext_min_z, ds_ext_max_z = ext_min_z, ext_max_z
        full_nx, full_ny, full_nz = vol_nx, vol_ny, vol_nz

    print(f"[syglass to imaris] IDataSet extents: X=[{ds_ext_min_x:.1f}, {ds_ext_max_x:.1f}] "
          f"Y=[{ds_ext_min_y:.1f}, {ds_ext_max_y:.1f}] Z=[{ds_ext_min_z:.1f}, {ds_ext_max_z:.1f}]")
    print(f"[syglass to imaris] ims/syk ratio: X={size_x/full_nx:.2f}  Y={size_y/full_ny:.2f}  Z={size_z/full_nz:.2f}")

    # Each label gets its own ISurfaces object so colors can be set independently.
    scene = vApp.GetSurpassScene()

    for label_idx, label_id in enumerate(label_ids):
        mask = (label_vol == label_id)
        print(f"[syglass to imaris] Building surface for label {label_id} "
              f"({int(mask.sum())} voxels, grid {vol_nx}x{vol_ny}x{vol_nz}) ...")

        # I initially put this in to deal with fin-like flanges off surface faces,
        # but that problem actually came from an off-by-one error in .syk parser.
        # This still does smooth out the mask a bit which is imo helpful.
        # 3 iterations of 7-neighbour averaging (~3 voxel radius)
        mask_f = _smooth_mask_3d(mask, iterations=3)

        sdf_ds = factory.CreateDataSet()
        sdf_ds.Create(eTypeUInt8, vol_nx, vol_ny, vol_nz, 1, 1)
        sdf_ds.SetExtendMinX(ds_ext_min_x); sdf_ds.SetExtendMinY(ds_ext_min_y); sdf_ds.SetExtendMinZ(ds_ext_min_z)
        sdf_ds.SetExtendMaxX(ds_ext_max_x); sdf_ds.SetExtendMaxY(ds_ext_max_y); sdf_ds.SetExtendMaxZ(ds_ext_max_z)

        for z in range(vol_nz):
            # sdf_z shape: (X, Y); values: 100=inside, 200=outside
            sdf_z = np.where(mask_f[:, :, z] > 0.5, 100, 200).astype(np.uint8)
            # tBytes2D is indexed [X][Y]: outer list length = sizeX, each bytes = Y strip
            slice_2d = [sdf_z[x, :].tobytes() for x in range(vol_nx)]
            sdf_ds.SetDataSliceBytes(slice_2d, z, 0, 0)
            if z % 100 == 0:
                print(f"[syglass to imaris]   label {label_id}: slice {z}/{vol_nz}")

        surfaces = factory.CreateSurfaces()
        surfaces.AddSurface(sdf_ds, 0)
        print(f"[syglass to imaris]   label {label_id}: AddSurface done")

        r, g, b = _DEFAULT_PALETTE[(label_id - 1) % len(_DEFAULT_PALETTE)]
        surfaces.SetColorRGBA(_pack_rgba(r, g, b, 255))
        surfaces.SetName(f"{ims_stem} label {label_id}")

        com = surfaces.GetCenterOfMass(0)
        try:
            surf_area = surfaces.GetArea(0)
            surf_vol  = surfaces.GetVolume(0)
            print(f"[syglass to imaris]   label {label_id}: CoM={com}  area={surf_area:.2f} um2  volume={surf_vol:.2f} um3")
        except Exception as _e:
            print(f"[syglass to imaris]   label {label_id}: CoM={com}  (GetArea/GetVolume failed: {_e})")

        scene.AddChild(surfaces, -1)

    print(f"[syglass to imaris] Created {len(label_ids)} surfaces.")

    # ----------------------------------------------------------------
    # 5. Import counting points (spots) from .sym if present
    # ----------------------------------------------------------------
    # TODO: Test this? Make a syglass project with points.
    if os.path.exists(sym_path):
        spots_xyz = _read_counting_points_from_sym(
            sym_path,
            syk_path=syk_path,
            ext_min=np.array([ext_min_x, ext_min_y, ext_min_z]),
            voxel_size=voxel_size,
        )
        if spots_xyz is not None and len(spots_xyz) > 0:
            vSpots = factory.CreateSpots()
            n = len(spots_xyz)
            radii     = [0.5] * n          # default 0.5 µm radius
            indices_t = [0]   * n          # all at t=0
            vSpots.Set(spots_xyz.tolist(), indices_t, radii)
            vSpots.SetName(ims_stem + " (syGlass counting points)")
            scene.AddChild(vSpots, -1)
            print(f"[syglass to imaris] Imported {n} counting points.")

    # ----------------------------------------------------------------
    # 6. Save
    # ----------------------------------------------------------------
    vApp.FileSave("", "")
    print("[syglass to imaris] Done — scene saved.")

# -----------------------------------------------------------------------
# Mask reader: manual .syk parser (fallback)
# -----------------------------------------------------------------------

def _read_mask_from_syk(
    syk_path: str,
    size_x: int,
    size_y: int,
    size_z: int,
):
    """
    Read the syGlass label mask from a .syk file.

    Scans all block headers to inventory the octree, reads the root block to
    estimate the mask bounding box, then composites only the deepest-level
    blocks that fall within that bbox.  This avoids allocating the full
    (potentially huge) reconstructed volume.

    Returns ((NX, NY, NZ) uint32 array, clip_info) where
      clip_info = (vx0, vy0, vz0, full_nx, full_ny, full_nz)
    or (None, None) on failure.
    """
    FVGU = b"fvgu"
    BLOCK_MARGIN = 2   # extra root-level blocks of margin on each side of bbox
    try:
        syk_size = os.path.getsize(syk_path)
        with open(syk_path, "rb") as f:
            header = f.read(36)
            if len(header) < 36:
                return None, None
            _, _, cx, cy, cz = struct.unpack_from("<5I", header, 0)

            block_payload = cx * cy * cz * 2
            block_stride  = 24 + block_payload
            n_blocks_est  = (syk_size - 36) // block_stride

            print(f"[syglass to imaris] .syk: cx={cx} cy={cy} cz={cz}  "
                  f"stride={block_stride:,}  ~{n_blocks_est} blocks")

            # Pass 1 — read all block headers (no payload) to inventory IDs
            block_offset = {}   # block_id → file offset
            off = 36
            for _ in range(n_blocks_est):
                f.seek(off)
                hdr = f.read(24)
                if len(hdr) < 24 or hdr[0:4] != FVGU:
                    break
                pl_size, bid, _lod, _flags = struct.unpack("<QIII", hdr[4:])
                if pl_size == block_payload:
                    block_offset[bid] = off
                off += block_stride

            if not block_offset:
                print("[syglass to imaris] .syk: no valid blocks found")
                return None, None

            max_level = max(_syk_block_level(bid) for bid in block_offset)
            n_grid    = 2 ** max_level
            # X blocks have a ghost slice at their high end (n_grid*(cx-1) matches .ims X).
            # Y and Z have no ghost slices.
            full_nx, full_ny, full_nz = n_grid * (cx - 1), n_grid * cy, n_grid * cz

            print(f"[syglass to imaris] .syk: {len(block_offset)} blocks; "
                  f"deepest level={max_level}  full grid={full_nx}×{full_ny}×{full_nz}")

            # Read root block to estimate mask bbox in root voxels, then scale up.
            # Use the full cx×cy×cz root block (ghost slices are fine here — they
            # just slightly widen the bbox estimate).
            bbox_x0, bbox_y0, bbox_z0 = 0, 0, 0
            bbox_x1, bbox_y1, bbox_z1 = full_nx, full_ny, full_nz
            if 0 in block_offset:
                f.seek(block_offset[0] + 24)
                raw_root = f.read(block_payload)
                root_arr = (np.frombuffer(raw_root, dtype="<u2")
                            .reshape(cz, cy, cx).transpose(2, 1, 0))
                nz_r = np.where(root_arr > 0)
                if len(nz_r[0]) > 0:
                    m = BLOCK_MARGIN
                    bbox_x0 = max(0,       (int(nz_r[0].min()) - m) * n_grid)
                    bbox_x1 = min(full_nx, (int(nz_r[0].max()) + m + 1) * n_grid)
                    bbox_y0 = max(0,       (int(nz_r[1].min()) - m) * n_grid)
                    bbox_y1 = min(full_ny, (int(nz_r[1].max()) + m + 1) * n_grid)
                    bbox_z0 = max(0,       (int(nz_r[2].min()) - m) * n_grid)
                    bbox_z1 = min(full_nz, (int(nz_r[2].max()) + m + 1) * n_grid)

        clip_nx = bbox_x1 - bbox_x0
        clip_ny = bbox_y1 - bbox_y0
        clip_nz = bbox_z1 - bbox_z0
        print(f"[syglass to imaris] .syk: allocating clipped vol {clip_nx}×{clip_ny}×{clip_nz} "
              f"(origin {bbox_x0},{bbox_y0},{bbox_z0})")

        vol = np.zeros((clip_nx, clip_ny, clip_nz), dtype=np.uint32)

        # Pass 2 — read ALL blocks, coarsest first so finer blocks overwrite coarser.
        # syGlass uses adaptive octree plus users may not paint with uniform res.
        with open(syk_path, "rb") as f:
            n_placed = 0
            for bid in sorted(block_offset, key=_syk_block_level):
                boff = block_offset[bid]
                _lv, ix, iy, iz = _syk_block_position(bid)
                # scale: fine-grid voxels per coarse voxel (1 at max_level, 2^k at coarser levels)
                scale = n_grid >> _lv

                bx0, bx1 = ix * (cx - 1) * scale, (ix + 1) * (cx - 1) * scale
                by0, by1 = iy * cy * scale,        (iy + 1) * cy * scale
                bz0, bz1 = iz * cz * scale,        (iz + 1) * cz * scale

                if bx1 <= bbox_x0 or bx0 >= bbox_x1: continue
                if by1 <= bbox_y0 or by0 >= bbox_y1: continue
                if bz1 <= bbox_z0 or bz0 >= bbox_z1: continue

                f.seek(boff + 24)
                raw = f.read(block_payload)
                # Drop the ghost slice on X only; Y and Z have no ghost slices.
                arr = (np.frombuffer(raw, dtype="<u2")
                       .reshape(cz, cy, cx)
                       .transpose(2, 1, 0)   # → (cx, cy, cz)
                       [:-1, :, :]            # drop X ghost → (cx-1, cy, cz)
                       .astype(np.uint32))

                # Upsample coarse blocks to fine-grid resolution so they fill the
                # correct number of fine voxels.  Finer blocks placed later overwrite.
                if scale > 1:
                    arr = np.repeat(np.repeat(np.repeat(
                        arr, scale, axis=0), scale, axis=1), scale, axis=2)
                # arr shape: (cx-1)*scale × cy*scale × cz*scale

                fine_nx = (cx - 1) * scale
                fine_ny = cy * scale
                fine_nz = cz * scale
                sx0 = max(0, bbox_x0 - bx0);  sx1 = min(fine_nx, bbox_x1 - bx0)
                sy0 = max(0, bbox_y0 - by0);  sy1 = min(fine_ny, bbox_y1 - by0)
                sz0 = max(0, bbox_z0 - bz0);  sz1 = min(fine_nz, bbox_z1 - bz0)

                dx0 = bx0 + sx0 - bbox_x0;  dx1 = dx0 + (sx1 - sx0)
                dy0 = by0 + sy0 - bbox_y0;  dy1 = dy0 + (sy1 - sy0)
                dz0 = bz0 + sz0 - bbox_z0;  dz1 = dz0 + (sz1 - sz0)

                vol[dx0:dx1, dy0:dy1, dz0:dz1] = arr[sx0:sx1, sy0:sy1, sz0:sz1]
                n_placed += 1

        n_fine = sum(1 for b in block_offset if _syk_block_level(b) == max_level)
        print(f"[syglass to imaris] .syk: placed {n_placed} blocks "
              f"({n_fine} at finest level {max_level}, {n_placed - n_fine} coarser)")


        clip_info = (bbox_x0, bbox_y0, bbox_z0, full_nx, full_ny, full_nz)
        return vol, clip_info

    except Exception as exc:
        print(f"[syglass to imaris] .syk parse failed: {exc}")
        if "Permission denied" in str(exc) or "13" in str(exc):
            print("[syglass to imaris] NOTE: .syk is locked by syGlass — direct file I/O impossible while syGlass holds it open. API path is required.")
        return None, None


def _syk_block_level(block_id: int) -> int:
    """Octree depth of a block given its breadth-first slot index."""
    if block_id == 0:
        return 0
    level, total, count = 0, 0, 1
    while total + count <= block_id:
        total += count
        count *= 8
        level += 1
    return level


def _syk_block_position(block_id: int):
    """Return (level, ix, iy, iz) for a block; child order: x=LSB, z=MSB."""
    if block_id == 0:
        return 0, 0, 0, 0
    parent_id  = (block_id - 1) // 8
    child_idx  = (block_id - 1) % 8
    x_bit = child_idx & 1
    y_bit = (child_idx >> 1) & 1
    z_bit = (child_idx >> 2) & 1
    p_lv, p_ix, p_iy, p_iz = _syk_block_position(parent_id)
    return p_lv + 1, 2 * p_ix + x_bit, 2 * p_iy + y_bit, 2 * p_iz + z_bit


# -----------------------------------------------------------------------
# Counting points reader from .sym
# -----------------------------------------------------------------------

def _read_counting_points_from_sym(
    sym_path: str,
    syk_path: str,
    ext_min: np.ndarray,
    voxel_size: np.ndarray,
) -> np.ndarray | None:
    """
    Read syGlass counting points from .sym (ZIP-wrapped LevelDB).

    Counting points are stored under keys 'default::countingPoints::N'.
    Each entry is 20 bytes — exact format TBD (confirm at some point).
    
    TODO: this is pure slop and must be whipped into shape.

    TODO: Confirm the binary format of each counting point record.
    Currently assuming XYZ in syGlass voxel space.

    Returns (N, 3) float32 in Imaris physical µm, or None.
    """
    try:
        try:
            import leveldb  # type: ignore
        except ImportError:
            print("[syglass to imaris] leveldb not available — cannot read .sym counting points.")
            return None

        # Read center_xyz from .syk header (bytes 8–19: center_x, center_y, center_z as uint32).
        # Coordinate transform: imaris_physical = (sg_coord + center) * voxel_size + ext_min
        center = np.zeros(3, dtype=np.float32)
        try:
            with open(syk_path, "rb") as f:
                hdr = f.read(20)
            _, _, cx, cy, cz = struct.unpack_from("<5I", hdr, 0)
            center = np.array([cx, cy, cz], dtype=np.float32)
        except Exception as exc:
            print(f"[syglass to imaris] WARNING: could not read .syk center; coordinate transform will be wrong: {exc}")

        tmpdir = tempfile.mkdtemp(prefix="pulse_xt_")
        try:
            with zipfile.ZipFile(sym_path, "r") as zf:
                zf.extractall(tmpdir)

            db = leveldb.LevelDB(tmpdir)
            points = []

            # Iterate keys matching 'default::countingPoints::\d+'
            prefix = b"default::countingPoints::"
            for key, val in db.RangeIter():
                if not key.startswith(prefix):
                    continue
                # TODO: Confirm 20-byte record format.
                # Hypothesis: 5 float32s (x, y, z, r, ?), syGlass voxel space.
                if len(val) >= 12:
                    x_sg, y_sg, z_sg = struct.unpack_from("<3f", val, 0)
                    sg_xyz = np.array([x_sg, y_sg, z_sg])
                    phys_xyz = (sg_xyz + center) * voxel_size + ext_min
                    points.append(phys_xyz)

            del db
            if points:
                return np.array(points, dtype=np.float32)
            return None

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    except Exception as exc:
        print(f"[syglass to imaris] .sym counting points read failed: {exc}")
        return None


# -----------------------------------------------------------------------
# Mask smoothing
# -----------------------------------------------------------------------

def _smooth_mask_3d(mask, iterations=3):
    """
    Smooth a binary mask by iterative 7-neighbour averaging.

    Each pass replaces every voxel with the average of itself and its 6
    face-neighbours, then we threshold at 0.5 to keep the field continuous.
    After 'iterations' passes the effective smoothing radius is ~iterations
    voxels, which rounds flat faces into curved caps without significantly 
    displacing the surface.

    Returns a float32 array in [0, 1]; caller should threshold at 0.5.
    """
    m = mask.astype(np.float32)
    for _ in range(iterations):
        n = m.copy()
        n[1:]    += m[:-1];   n[:-1]   += m[1:]
        n[:, 1:] += m[:, :-1]; n[:, :-1] += m[:, 1:]
        n[:, :, 1:] += m[:, :, :-1]; n[:, :, :-1] += m[:, :, 1:]
        m = n / 7.0
    return m


# -----------------------------------------------------------------------
# Colour helpers
# -----------------------------------------------------------------------

_DEFAULT_PALETTE = [
    (255, 0,   0  ),  # 1 = red
    (128, 255, 0  ),  # 2 = chartreuse
    (0,   255, 255),  # 3 = cyan
    (128, 0,   255),  # 4 = violet
    (255, 112, 64 ),  # 5 = coral
    (112, 255, 64 ),  # 6 = yellow-green
    (64,  208, 255),  # 7 = sky blue
    (208, 64,  255),  # 8 = purple
    (255, 192, 128),  # 9 = peach
    (128, 255, 128),  # 10 = mint
    (128, 192, 255),  # 11 = periwinkle
    (255, 128, 255),  # 12 = pink
    (255, 192, 0  ),  # 13 = amber
    (0,   255, 64 ),  # 14 = spring green
    (0,   64,  255),  # 15 = blue
    (255, 0,   192),  # 16 = hot pink
]


def _pack_rgba(r: int, g: int, b: int, a: int) -> int:
    """Pack RGBA uint8 values into a signed int32 (Imaris format)."""
    val = (a << 24) | (b << 16) | (g << 8) | r
    # Convert to signed int32 (Imaris uses signed RGBA)
    if val >= 0x80000000:
        val -= 0x100000000
    return val


# -----------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------
# TODO: make this work good and ideally open a window in Imaris
def _log_warning(vApp, message: str) -> None:
    """Show a warning dialog and print to stdout."""
    print(f"[syglass to imaris] WARNING: {message}")
    try:
        import tkinter.messagebox
        tkinter.messagebox.showwarning("Pulse", message)
    except Exception:
        pass
