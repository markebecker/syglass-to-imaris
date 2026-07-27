"""
Imaris XTension: Import syGlass Masks and Counting Points into Imaris.

Runs inside a live Imaris session (Windows only).  Imaris calls
XTImportFromSyGlass(imarisFile) where imarisFile is the numeric handle
for the ImarisLib COM connection.

Usage
-----
1. Open the .ims file in Imaris.
2. Run Extensions → Import from syGlass.  The matching .syk is found automatically
   if it sits beside the .ims; otherwise a file picker opens so it can be anywhere.

Strategy
--------
Masks:
  1. Parse the .syk root block (header + block 0 payload) directly, rather than using `syglass` Python API. (This has proven to be more trouble than it sees like it would be).
     The root block stores a 2x-downsampled label volume in (CZ, CY, CX) order.
  2. For each label, build a signed uint8 field (100=inside, 200=outside) cropped to the
     label's bounding box, upload it in z-band chunks via SetDataSubVolumeAs1DArrayBytes
     (kept under the Ice message limit),
  3. then call ISurfaces.AddSurface once — Imaris meshes
     the zero crossing for all the label's blobs.  Prep for the next label runs on a worker
     thread while the current one uploads.

Progress + logs:
  A tkinter window shows the estimated finish time.  A timestamped log is written to a
  'logs' folder parallel to this script's 'core' folder (…/<shuttle>/logs/pulse_xt_*.log).

Spots (counting points):
  Read countingPoints from .sym LevelDB, convert coordinates back to Imaris
  physical space, create ISpots.  (Stub — format confirmation pending.)

Python 3.11 compatibility required (Imaris bundled interpreter).
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import struct
import sys
import tempfile
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# ImarisLib — always available in Imaris's Python interpreter
import ImarisLib  # type: ignore  # noqa: E402  (Windows COM wrapper)
# NOTE: the syglass/pyglass API path was removed.  pyglass is geared toward .syg-based
# projects and did not work well with the .syk/.ims pairs used here, so masks come straight
# from the .syk parser (_read_mask_from_syk).

# Largest payload (bytes) to push in a single COM call.  Ice's default MessageSizeMax
# is ~1 MB; staying under it lets us upload a band of z-slices per call.
_ICE_CHUNK_BYTES = 900_000

# Remembers the last-used .syk directory for the file picker.
_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".pulse_xt.json")

# Open log file handle (set by _setup_logging); every _log() line carries a timestamp
# so it is clear which step consumes time.  None until setup, in which case _log()
# still prints to the console.
_LOG_FH = None


def _setup_logging():
    """
    Open a timestamped log file in a 'logs' folder INSIDE the script's own directory —
    i.e. <script_dir>/logs/pulse_xt_<timestamp>.log.  The script lives at the root of the
    git repo, so the log lands inside the working tree and can be committed + pushed to
    share it (the network-drive shuttle is unavailable).  Returns the path, or None on failure.
    """
    global _LOG_FH
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir  = os.path.join(script_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        stamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"pulse_xt_{stamp}.log")
        _LOG_FH  = open(log_path, "a", buffering=1)  # line-buffered
        return log_path
    except Exception as exc:
        _LOG_FH = None
        _log(f"[syglass to imaris] could not open log file ({exc}); console only.")
        return None


def _log(msg: str) -> None:
    """Print a timestamped line to the console and, if open, the log file."""
    line = f"{datetime.datetime.now().strftime('%H:%M:%S')} {msg}"
    print(line)
    if _LOG_FH is not None:
        try:
            _LOG_FH.write(line + "\n")
        except Exception:
            pass


# -----------------------------------------------------------------------
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
        # Try several paths — at least one should be writable
        for log_path in [
            os.path.join(os.path.expanduser("~"), "pulse_xt_crash.log"),
            r"C:\pulse_xt_crash.log",
            r"D:\pulse_xt_crash.log",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "pulse_xt_crash.log"),
        ]:
            try:
                with open(log_path, "a") as f:
                    f.write(tb)
                _log(f"[syglass to imaris] CRASH — traceback written to: {log_path}")
                break
            except Exception:
                pass
        print(tb)  # also try stdout in case console is still open


def _run(imarisFile: int) -> None:
    """Inner implementation (wrapped for crash logging)."""
    log_path = _setup_logging()
    if log_path:
        _log(f"[syglass to imaris] logging to: {log_path}")

    vLib = ImarisLib.ImarisLib()
    vApp = vLib.GetApplication(imarisFile)

    # ----------------------------------------------------------------
    # 1. Locate companion syGlass files
    # ----------------------------------------------------------------
    ims_path = vApp.GetCurrentFileName()
    if not ims_path:
        _log_warning(vApp, "No file is currently open in Imaris.")
        return

    ims_dir  = os.path.dirname(ims_path)
    ims_stem = os.path.splitext(os.path.basename(ims_path))[0]

    # Locate the .syk: try the conventional same-dir/same-stem path first, then a
    # case-insensitive scan, then fall back to a GUI file picker (the .syk may live
    # elsewhere or be named differently from the .ims).
    syk_path = _resolve_syk_path(ims_dir, ims_stem)
    if not syk_path or not os.path.exists(syk_path):
        _log_warning(vApp, "No .syk file selected — aborting import.")
        return

    # The .sym (counting points / metadata) sits next to the chosen .syk, sharing its stem.
    syk_dir  = os.path.dirname(syk_path)
    syk_stem = os.path.splitext(os.path.basename(syk_path))[0]
    sym_path = os.path.join(syk_dir, syk_stem + ".sym")
    _log(f"[syglass to imaris] using .syk: {syk_path}")

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
    _log(f"[syglass to imaris] Importing your syglass mask...")
    _log(f"[syglass to imaris] .ims dims:    {size_x} x {size_y} x {size_z} voxels")
    _log(f"[syglass to imaris] .ims extents: X=[{ext_min_x:.3f}, {ext_max_x:.3f}] "
          f"Y=[{ext_min_y:.3f}, {ext_max_y:.3f}] Z=[{ext_min_z:.3f}, {ext_max_z:.3f}]")
    _log(f"[syglass to imaris] .ims voxel size (um): {voxel_size[0]:.4f} x {voxel_size[1]:.4f} x {voxel_size[2]:.4f}")

    # ----------------------------------------------------------------
    # 3. Read syGlass label mask
    # ----------------------------------------------------------------
    label_vol = None  # will be (NX, NY, NZ) uint32, 0=background, 1..N=label

    clip_info = None   # (vx0,vy0,vz0, full_nx,full_ny,full_nz) or None

    # Progress window is created here (before the multi-second read) so the user sees each
    # phase; it is closed in the finally at the end of the surface loop.
    prog = _Progress("syGlass → Imaris — importing mask")

    prog.phase("Reading syGlass mask from .syk")
    t_read0 = time.time()
    label_vol, clip_info = _read_mask_from_syk(syk_path, size_x, size_y, size_z)

    if label_vol is None or not np.any(label_vol):
        prog.close()
        _log_warning(vApp, "No mask data found in .syk file (or file is empty).")
        return
    _log(f"[syglass to imaris] mask read in {time.time() - t_read0:.1f}s")

    # Inventory labels in a SINGLE pass with bincount (np.unique sorts the whole 7-billion
    # voxel array — ~70s here; bincount is one O(n) scan).  Bins index label value directly,
    # so counts come for free.  Implausible IDs (>100, e.g. stray 18xxx/31xxx) are simply
    # not processed — no need to zero them out of the volume (another full pass avoided).
    prog.phase("Inventorying labels")
    t_scan0 = time.time()
    counts = np.bincount(label_vol.ravel())
    all_ids = [int(v) for v in np.nonzero(counts)[0] if v != 0]
    label_ids = [v for v in all_ids if v <= 100]
    ignored   = [v for v in all_ids if v > 100]
    nonzero = int(label_vol.size - counts[0])
    _log(f"[syglass to imaris] label_vol shape: {label_vol.shape}, dtype: {label_vol.dtype}")
    _log(f"[syglass to imaris] Non-zero voxels: {nonzero} / {label_vol.size}  "
         f"(inventory in {time.time() - t_scan0:.1f}s)")
    for v in all_ids:
        _log(f"[syglass to imaris]   label {v}: {int(counts[v])} voxels")
    if ignored:
        _log(f"[syglass to imaris] Ignoring {len(ignored)} implausible label IDs (>100): {ignored}")
    _log(f"[syglass to imaris] Found {len(label_ids)} labels: {label_ids}")
    prog.start_labels(len(label_ids))

    # ASCII max-projections of the reconstructed mask so its actual shape is visible in the
    # log (independent of Imaris) — useful for telling poor reconstruction from mask issues.
    prog.phase("Rendering mask preview")
    _log_mask_projections(label_vol)

    # ----------------------------------------------------------------
    # 4. Build one ISurfaces per label via AddSurface.
    #
    #    AddSurface interprets the dataset as a signed field and finds the surface at the
    #    zero crossing.  uint8 is treated as signed int8: 100 → +100 (inside), 200 → -56
    #    (outside).  Each label's IDataSet is sized to that label's OWN bounding box (not
    #    the full reconstruction — which may be billions of mostly-empty voxels), with
    #    physical extents for just that sub-region.  Data is uploaded in chunks kept under
    #    the Ice message limit (see _upload_field).
    # ----------------------------------------------------------------
    factory = vApp.GetFactory()

    # Resolve eTypeUInt8 — the enum lives in ImarisLib or a submodule
    eTypeUInt8 = None
    # Try 1: ImarisLib module attribute (in Python 3.7 bundled interpreter)
    if hasattr(ImarisLib, 'tType'):
        eTypeUInt8 = ImarisLib.tType.eTypeUInt8
    if eTypeUInt8 is None:
        # Try 2: scan loaded modules for one that exposes tType
        for _mod in list(sys.modules.values()):
            if _mod is not None and hasattr(_mod, 'tType'):
                eTypeUInt8 = _mod.tType.eTypeUInt8
                break
    if eTypeUInt8 is None:
        # Try 3: use integer value directly (eTypeUInt8 = 0 in all known Imaris versions)
        _log("[syglass to imaris] WARNING: could not resolve tType enum; using raw integer 0 for eTypeUInt8")
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
        _log(f"[syglass to imaris] IDataSet (clipped): {vol_nx}×{vol_ny}×{vol_nz} voxels  "
              f"full grid {full_nx}×{full_ny}×{full_nz}")
    else:
        ds_ext_min_x, ds_ext_max_x = ext_min_x, ext_max_x
        ds_ext_min_y, ds_ext_max_y = ext_min_y, ext_max_y
        ds_ext_min_z, ds_ext_max_z = ext_min_z, ext_max_z
        full_nx, full_ny, full_nz = vol_nx, vol_ny, vol_nz

    _log(f"[syglass to imaris] IDataSet extents: X=[{ds_ext_min_x:.1f}, {ds_ext_max_x:.1f}] "
          f"Y=[{ds_ext_min_y:.1f}, {ds_ext_max_y:.1f}] Z=[{ds_ext_min_z:.1f}, {ds_ext_max_z:.1f}]")
    _log(f"[syglass to imaris] ims/syk ratio: X={size_x/full_nx:.2f}  Y={size_y/full_ny:.2f}  Z={size_z/full_nz:.2f}")

    # Physical size of one clip voxel — converts each label's crop (in clip-voxel coords)
    # into the µm extents of its own IDataSet.
    vx_clip = (ds_ext_max_x - ds_ext_min_x) / vol_nx
    vy_clip = (ds_ext_max_y - ds_ext_min_y) / vol_ny
    vz_clip = (ds_ext_max_z - ds_ext_min_z) / vol_nz

    # Each label gets its own ISurfaces object so colors can be set independently.
    scene = vApp.GetSurpassScene()
    n_labels = len(label_ids)

    # Prefetch depth 1: prep the NEXT label (crop + smooth + SDF build) on a background thread
    # while the main thread uploads the CURRENT one.  Prep is pure NumPy (releases the GIL) so
    # it overlaps the serial COM I/O; depth 1 caps concurrent prep RAM.
    _log(f"[syglass to imaris] {n_labels} label(s); each cropped to its bbox, "
         f"uploaded in <= {_ICE_CHUNK_BYTES // 1000} KB chunks; prep pipelined 1 ahead")

    t_run0 = time.time()
    sum_prep = sum_upload = sum_detect = 0.0

    try:
        # max_workers=2 is enough for depth-1 pipelining and bounds concurrent prep RAM.
        with ThreadPoolExecutor(max_workers=2) as pool:
            next_future = pool.submit(_prep_label, label_vol, label_ids[0])
            for label_idx, label_id in enumerate(label_ids):
                tag = f"L{label_idx + 1}/{n_labels} (id {label_id})"

                prog.preparing(label_idx)      # caption shown during the (blocking) prep wait
                t0 = time.time()
                prep = next_future.result()
                t_prep = time.time() - t0
                # Kick off prep for the next label so it overlaps this label's upload.
                if label_idx + 1 < n_labels:
                    next_future = pool.submit(_prep_label, label_vol, label_ids[label_idx + 1])

                if prep is None:
                    _log(f"[syglass to imaris] {tag}: empty — skipped")
                    prog.begin_label(label_idx, 1); prog.end_label(0.0)
                    continue

                field, (ox, oy, oz), n_painted, n_kept = prep
                cnx, cny, cnz = field.shape
                prog.begin_label(label_idx, _upload_ticks(cnx, cny, cnz))

                # Physical extents of this label's crop (sub-region of the clip volume).
                sx0 = ds_ext_min_x + ox * vx_clip;  sx1 = ds_ext_min_x + (ox + cnx) * vx_clip
                sy0 = ds_ext_min_y + oy * vy_clip;  sy1 = ds_ext_min_y + (oy + cny) * vy_clip
                sz0 = ds_ext_min_z + oz * vz_clip;  sz1 = ds_ext_min_z + (oz + cnz) * vz_clip
                mb = cnx * cny * cnz / 1e6
                _log(f"[syglass to imaris] {tag}: crop {cnx}x{cny}x{cnz} ({mb:.1f} MB) at voxel "
                     f"({ox},{oy},{oz}); painted={n_painted} kept={n_kept}; "
                     f"bbox um X=[{sx0:.0f},{sx1:.0f}] Y=[{sy0:.0f},{sy1:.0f}] Z=[{sz0:.0f},{sz1:.0f}]")

                sdf_ds = factory.CreateDataSet()
                sdf_ds.Create(eTypeUInt8, cnx, cny, cnz, 1, 1)
                sdf_ds.SetExtendMinX(sx0); sdf_ds.SetExtendMinY(sy0); sdf_ds.SetExtendMinZ(sz0)
                sdf_ds.SetExtendMaxX(sx1); sdf_ds.SetExtendMaxY(sy1); sdf_ds.SetExtendMaxZ(sz1)

                t0 = time.time()
                _upload_field(sdf_ds, field, prog)
                t_up = time.time() - t0
                del field, prep   # free this label's prep before the next is awaited

                # ONE AddSurface on the whole cropped field — meshes all of the label's
                # blobs at once (the original working behaviour).
                surfaces = factory.CreateSurfaces()
                prog.detecting()
                t0 = time.time()
                surfaces.AddSurface(sdf_ds, 0)
                t_det = time.time() - t0
                try:
                    n_surf = surfaces.GetNumberOfSurfaces()
                except Exception as _e:
                    n_surf = f"?({_e})"

                r, g, b = _DEFAULT_PALETTE[(label_id - 1) % len(_DEFAULT_PALETTE)]
                surfaces.SetColorRGBA(_pack_rgba(r, g, b, 255))
                surfaces.SetName(f"{ims_stem} label {label_id}")
                scene.AddChild(surfaces, -1)

                sum_prep += t_prep; sum_upload += t_up; sum_detect += t_det
                _log(f"[syglass to imaris] {tag}: prep_wait={t_prep:.2f}s  "
                     f"upload={t_up:.2f}s  detect={t_det:.2f}s  surfaces_in_object={n_surf}")
                prog.end_label(t_prep + t_up + t_det)
    finally:
        prog.close()

    t_total = time.time() - t_run0
    _log(f"[syglass to imaris] Created {n_labels} surface object(s) in {t_total:.1f}s  "
         f"(prep_wait {sum_prep:.1f}s + upload {sum_upload:.1f}s + detect {sum_detect:.1f}s; "
         f"avg {t_total / max(1, n_labels):.2f}s/label)")

    # ----------------------------------------------------------------
    # 5. Import counting points (spots) from .sym if present
    # ----------------------------------------------------------------
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
            _log(f"[syglass to imaris] Imported {n} counting points.")

    # ----------------------------------------------------------------
    # 6. Done — intentionally NO save.
    #    vApp.FileSave("", "") pops Imaris's Save dialog (empty filename) and rewrites the
    #    whole .ims (~3 min on large files).  The surfaces are already in the live Surpass
    #    scene, so we leave saving to the user (Ctrl+S) whenever they choose.
    # ----------------------------------------------------------------
    _log("[syglass to imaris] Done — surfaces added to the scene. NOT saved: press Ctrl+S "
         "in Imaris to write them to the .ims when ready.")


# -----------------------------------------------------------------------
# Mask reader: manual .syk parser (main)
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

            _log(f"[syglass to imaris] .syk: cx={cx} cy={cy} cz={cz}  "
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
                _log("[syglass to imaris] .syk: no valid blocks found")
                return None, None

            max_level = max(_syk_block_level(bid) for bid in block_offset)
            n_grid    = 2 ** max_level
            # Adjacent blocks SHARE a 1-voxel apron (confirmed by the apron diagnostic:
            # block A's last slice ≈ block B's first slice).  So each block holds only (c-1)
            # UNIQUE voxels; the effective stride and full grid use (c-1), and n_grid root
            # voxels still map to n_grid×(c-1) full voxels (bbox scaling below stays correct).
            ux, uy, uz = cx - 1, cy - 1, cz - 1
            full_nx, full_ny, full_nz = n_grid * ux, n_grid * uy, n_grid * uz

            _log(f"[syglass to imaris] .syk: {len(block_offset)} blocks; "
                  f"deepest level={max_level}  full grid={full_nx}×{full_ny}×{full_nz}")

            # DIAGNOSTIC: are adjacent blocks overlapping (a shared ghost/apron slice)?
            # This tells us the correct block stride so the seam/ghost-slab artifact can be
            # fixed properly instead of guessed.
            _diagnose_block_apron(f, block_offset, cx, cy, cz, max_level, block_payload)

            # Read root block to estimate mask bbox in root voxels, then scale up.
            # Use the full cx×cy×cz root block (ghost slices are fine here — they
            # just slightly widen the bbox estimate, which is conservative).
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
        _log(f"[syglass to imaris] .syk: allocating clipped vol {clip_nx}×{clip_ny}×{clip_nz} "
              f"(origin {bbox_x0},{bbox_y0},{bbox_z0})")

        # uint16 (not uint32): the .syk payload is already uint16 and label IDs fit, so
        # this halves the memory bandwidth of the seam fill and every downstream scan.
        vol = np.zeros((clip_nx, clip_ny, clip_nz), dtype=np.uint16)

        # Pass 2 — read max_level blocks within bbox and place them
        with open(syk_path, "rb") as f:
            n_placed = 0
            for bid, boff in block_offset.items():
                if _syk_block_level(bid) != max_level:
                    continue
                _lv, ix, iy, iz = _syk_block_position(bid)

                bx0, bx1 = ix * ux, ix * ux + ux    # stride (c-1): neighbours share an apron
                by0, by1 = iy * uy, iy * uy + uy
                bz0, bz1 = iz * uz, iz * uz + uz

                if bx1 <= bbox_x0 or bx0 >= bbox_x1: continue
                if by1 <= bbox_y0 or by0 >= bbox_y1: continue
                if bz1 <= bbox_z0 or bz0 >= bbox_z1: continue

                f.seek(boff + 24)
                raw = f.read(block_payload)
                # Drop the shared apron (last slice on each axis) and place the (c-1) UNIQUE
                # voxels at stride (c-1).  The neighbour block supplies that boundary slice as
                # its OWN first slice, so the volume tiles seamlessly — no gaps, no ghost slabs.
                arr = (np.frombuffer(raw, dtype="<u2")
                       .reshape(cz, cy, cx)[:-1, :-1, :-1]
                       .transpose(2, 1, 0))   # (cx-1, cy-1, cz-1), uint16

                sx0 = max(0, bbox_x0 - bx0);  sx1 = min(ux, bbox_x1 - bx0)
                sy0 = max(0, bbox_y0 - by0);  sy1 = min(uy, bbox_y1 - by0)
                sz0 = max(0, bbox_z0 - bz0);  sz1 = min(uz, bbox_z1 - bz0)

                dx0 = bx0 + sx0 - bbox_x0;  dx1 = dx0 + (sx1 - sx0)
                dy0 = by0 + sy0 - bbox_y0;  dy1 = dy0 + (sy1 - sy0)
                dz0 = bz0 + sz0 - bbox_z0;  dz1 = dz0 + (sz1 - sz0)

                vol[dx0:dx1, dy0:dy1, dz0:dz1] = arr[sx0:sx1, sy0:sy1, sz0:sz1]
                n_placed += 1

        _log(f"[syglass to imaris] .syk: placed {n_placed} level-{max_level} blocks "
              f"(of {n_grid**3} possible)")

        # The X block boundary leaves a 1-voxel gap (seam probe: '111111.|11111111'); Y/Z
        # tile cleanly.  Fill any 0 voxel flanked by the SAME label on both sides — patches
        # the seams without inventing data at mask edges (edges have a 0 on one side).
        vol = _fill_block_seams(vol)
        _log("[syglass to imaris] .syk: block-boundary seams filled")

        # DIAGNOSTIC (after the fill): the profiles should now read '1111|1111' with no gap.
        _probe_seams(vol, ux, uy, uz, (bbox_x0, bbox_y0, bbox_z0))

        clip_info = (bbox_x0, bbox_y0, bbox_z0, full_nx, full_ny, full_nz)
        return vol, clip_info

    except Exception as exc:
        _log(f"[syglass to imaris] .syk parse failed: {exc}")
        if "Permission denied" in str(exc) or "13" in str(exc):
            _log("[syglass to imaris] NOTE: .syk is locked by syGlass — direct file I/O impossible while syGlass holds it open. API path is required.")
        return None, None


def _fill_block_seams(vol: np.ndarray) -> np.ndarray:
    """
    Fill 1-voxel seams left by dropped ghost slices at block boundaries.

    For each axis, any zero voxel that has the same nonzero label on both
    sides gets that label.  This patches the thin gaps mostly without adding
    data that wasn't in the original painting.
    """
    for axis in range(3):
        sl = [slice(None), slice(None), slice(None)]
        sc = [slice(None), slice(None), slice(None)]
        sr = [slice(None), slice(None), slice(None)]
        sl[axis] = slice(None, -2)
        sc[axis] = slice(1, -1)
        sr[axis] = slice(2, None)
        left   = vol[tuple(sl)]
        center = vol[tuple(sc)]
        right  = vol[tuple(sr)]
        fill   = (left == right) & (left > 0) & (center == 0)
        vol[tuple(sc)] = np.where(fill, left, center)
    return vol


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


def _probe_seams(vol, ux, uy, uz, origin, n_probes=4, half=7):
    """
    Log the reconstructed value profile across block boundaries, per axis, so seams / ghost
    slabs are directly visible in the log.  For each of a few boundaries it finds a transverse
    line where the mask is present and prints ±`half` voxels with '|' marking the boundary:
      '111|111' clean join · '11.|.11' or '11..11' gap/seam · misaligned = shift/ghost.
    Dependency-free; reads only a handful of lines.
    """
    ox0, oy0, oz0 = origin
    nx, ny, nz = vol.shape

    def _render(prof, bpos_rel):
        s = ""
        for j, v in enumerate(prof):
            if j == bpos_rel:
                s += "|"
            s += "." if v == 0 else (str(v) if v < 10 else "#")
        return s

    _log("[syglass to imaris] SEAM PROBE (values across block boundaries; | = block edge):")
    for axis, name, u, o, n in ((0, "X", ux, ox0, nx), (1, "Y", uy, oy0, ny), (2, "Z", uz, oz0, nz)):
        others = [i for i in range(3) if i != axis]
        # global block boundaries are at multiples of u; convert to clip coords
        bnds = [k * u - o for k in range(1, (o + n) // u + 2) if 0 < k * u - o < n]
        shown = 0
        for c in bnds:
            if shown >= n_probes:
                break
            lo, hi = max(0, c - half), min(n, c + half + 1)
            sl = [slice(None)] * 3
            sl[axis] = slice(lo, hi)
            band = vol[tuple(sl)]                      # (..slab..) over the two other axes
            nz2d = np.any(band > 0, axis=axis)
            hits = np.argwhere(nz2d)
            if len(hits) == 0:
                continue
            a, b = hits[len(hits) // 2]                # a transverse line with mask near the edge
            idx = [0, 0, 0]
            idx[others[0]] = int(a)
            idx[others[1]] = int(b)
            prof = []
            for x in range(lo, hi):
                idx[axis] = x
                prof.append(int(vol[tuple(idx)]))
            _log(f"[syglass to imaris]   {name}@{c}: {_render(prof, c - lo)}")
            shown += 1
        if shown == 0:
            _log(f"[syglass to imaris]   {name}: no painted block boundary found")


def _diagnose_block_apron(f, block_offset, cx, cy, cz, max_level, block_payload):
    """
    Determine the block-overlap structure per axis, AGGREGATED over ALL adjacent painted
    block pairs (a single pair is too often background).  For pair (A at i, B at i+1) sharing
    a boundary, it measures painted-voxel agreement of:
      high = A.last-slice  vs B.first-slice   (high-side apron: A's last duplicates boundary)
      low  = A.first-slice vs (prev)P.last    (low-side apron:  A's first duplicates boundary)
      base = A.mid-slice   vs B.mid-slice     (baseline for a smoothly-varying volume)
    Interpretation: high≫base ⇒ drop A's LAST slice; low≫base ⇒ drop A's FIRST slice;
    both≈base ⇒ blocks abut, drop nothing.  Reads each block once (cached).
    """
    pos2bid = {}
    for bid in block_offset:
        if _syk_block_level(bid) != max_level:
            continue
        _lv, ix, iy, iz = _syk_block_position(bid)
        pos2bid[(ix, iy, iz)] = bid

    cache = {}
    def _read(bid):
        if bid not in cache:
            f.seek(block_offset[bid] + 24)
            cache[bid] = np.frombuffer(f.read(block_payload), dtype="<u2").reshape(cz, cy, cx)
        return cache[bid]

    def _acc(a, b, tot):
        m = (a > 0) | (b > 0)
        return (tot[0] + int(np.sum(a[m] == b[m])), tot[1] + int(m.sum()))

    _log("[syglass to imaris] APRON DIAGNOSTIC (aggregated over all adjacent painted pairs):")
    for axis, name, dim in ((2, "X", cx), (1, "Y", cy), (0, "Z", cz)):
        unit = {2: (1, 0, 0), 1: (0, 1, 0), 0: (0, 0, 1)}[axis]
        hi = lo = base = (0, 0)
        npairs = 0
        for pos, bid in pos2bid.items():
            nb = tuple(p + u for p, u in zip(pos, unit))
            if nb in pos2bid:
                A, B = _read(bid), _read(pos2bid[nb])
                hi   = _acc(np.take(A, dim - 1, axis=axis), np.take(B, 0, axis=axis), hi)
                base = _acc(np.take(A, dim // 2, axis=axis), np.take(B, dim // 2, axis=axis), base)
                npairs += 1
            pb = tuple(p - u for p, u in zip(pos, unit))
            if pb in pos2bid:
                A, P = _read(bid), _read(pos2bid[pb])
                lo = _acc(np.take(A, 0, axis=axis), np.take(P, dim - 1, axis=axis), lo)
        def _r(t):
            return (t[0] / t[1]) if t[1] else float("nan")
        _log(f"[syglass to imaris]   {name}: high(A.last=B.first)={_r(hi):.2f}  "
             f"low(A.first=prev.last)={_r(lo):.2f}  base={_r(base):.2f}  "
             f"pairs={npairs} vox={hi[1]}")


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

    TODO: Confirm the binary format of each counting point record.
    Current assumption: XYZ in syGlass voxel space.

    Returns (N, 3) float32 in Imaris physical µm, or None.
    """
    try:
        try:
            import leveldb  # type: ignore
        except ImportError:
            _log("[syglass to imaris] leveldb not available — cannot read .sym counting points.")
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
            _log(f"[syglass to imaris] WARNING: could not read .syk center; coordinate transform will be wrong: {exc}")

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
        _log(f"[syglass to imaris] .sym counting points read failed: {exc}")
        return None


# -----------------------------------------------------------------------
# Mask smoothing
# -----------------------------------------------------------------------

def _prep_label(label_vol, label_id, iterations=3, margin=4):
    """
    Build the signed uint8 field AddSurface consumes for ONE label, CROPPED to that label's
    bounding box (CPU-only, thread-safe — no COM/Ice access).

    Returns (field, (ox,oy,oz), n_painted, n_kept) or None if the label is empty:
      field : (cnx,cny,cnz) uint8 crop, 100=inside / 200=outside.  Imaris reads uint8 as
              signed int8 (100→+100, 200→−56) and meshes the zero crossing.
      ox,oy,oz : crop origin in clip-voxel coords (caller sets the crop's µm extents).

    Cropping is the speed/memory win: the reconstruction can be billions of voxels while a
    label occupies a tiny fraction, so we smooth and upload only the bbox.  A `margin` of
    outside voxels is left around the box so the smoothed field returns to "outside" and the
    surface closes inside the crop.  Smoothing is UNION-ed with the original voxels so nothing
    painted is ever eroded.

    NOTE: this relies on AddSurface honouring the cropped IDataSet's SetExtendMin/Max — to be
    validated on a known-good .syk (a corrupted .syk previously masked whether this works).
    """
    mask = label_vol == label_id
    xs = np.any(mask, axis=(1, 2))
    if not xs.any():
        return None
    ys = np.any(mask, axis=(0, 2))
    zs = np.any(mask, axis=(0, 1))

    def _span(flags, n):
        idx = np.nonzero(flags)[0]
        return max(0, int(idx[0]) - margin), min(n, int(idx[-1]) + 1 + margin)

    ox, x1 = _span(xs, mask.shape[0])
    oy, y1 = _span(ys, mask.shape[1])
    oz, z1 = _span(zs, mask.shape[2])

    sub = mask[ox:x1, oy:y1, oz:z1].copy()   # small crop; copy so the big bool can free
    del mask
    n_painted = int(sub.sum())

    keep = (_smooth_mask_3d(sub, iterations=iterations) > 0.5) | sub   # never erode
    field = np.where(keep, 100, 200).astype(np.uint8)
    return field, (ox, oy, oz), n_painted, int(keep.sum())


def _ascii_projection(occ_2d, cols=64, rows=26):
    """Render a boolean 2D occupancy map as ASCII (block-max downsample) for the log."""
    h, w = occ_2d.shape
    yb = np.linspace(0, h, rows + 1).astype(int)
    xb = np.linspace(0, w, cols + 1).astype(int)
    lines = []
    for r in range(rows):
        if yb[r + 1] <= yb[r]:
            continue
        row = occ_2d[yb[r]:yb[r + 1]]
        line = "".join("#" if row[:, xb[c]:xb[c + 1]].any() else " "
                       for c in range(cols) if xb[c + 1] > xb[c])
        lines.append(line)
    return lines


def _log_mask_projections(label_vol):
    """Log ASCII max-projections of the reconstructed mask so its shape is visible."""
    occ = label_vol > 0
    top = np.any(occ, axis=2)   # (X,Y) — top-down footprint
    side = np.any(occ, axis=1)  # (X,Z) — side view
    _log("[syglass to imaris] mask projection — top-down (rows=X, cols=Y):")
    for ln in _ascii_projection(top):
        _log("[syglass to imaris]   |" + ln + "|")
    _log("[syglass to imaris] mask projection — side (rows=X, cols=Z):")
    for ln in _ascii_projection(side):
        _log("[syglass to imaris]   |" + ln + "|")


def _smooth_mask_3d(mask, iterations=3):
    """
    Smooth a binary mask by iterative 7-neighbour averaging.

    Each pass replaces every voxel with the average of itself and its 6
    face-neighbours, then we threshold at 0.5 to keep the field continuous.
    After 'iterations' passes the effective smoothing radius is ~iterations
    voxels, which rounds sharp block-boundary flat faces into curved caps
    without significantly displacing the surface.

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
# Upload to an IDataSet
# -----------------------------------------------------------------------

def _upload_ticks(nx: int, ny: int, nz: int, budget: int = _ICE_CHUNK_BYTES) -> int:
    """Number of COM calls _upload_field will make for an (nx,ny,nz) uint8 field."""
    slice_bytes = nx * ny
    if slice_bytes <= budget:
        zb = max(1, budget // slice_bytes)
        return (nz + zb - 1) // zb
    return nz   # one call per z-slice (each slice is itself Y-tiled)


def _upload_field(ds, field: np.ndarray, prog) -> int:
    """
    Upload a cropped (nx,ny,nz) uint8 field to an IDataSet via SetDataSubVolumeAs1DArrayBytes,
    keeping every COM call under the Ice message budget.

    Imaris' 1D layout is X-fastest, then Y, then Z.  field is (X,Y,Z), so transposing a
    sub-block to (Z,Y,X) and calling the default C-order .tobytes() yields x-fastest bytes.
    Ticks `prog` once per COM call.  Returns the number of calls.
    """
    nx, ny, nz = field.shape
    budget = _ICE_CHUNK_BYTES
    slice_bytes = nx * ny
    calls = 0
    if slice_bytes <= budget:
        zb = max(1, budget // slice_bytes)          # whole z-bands per call
        for z0 in range(0, nz, zb):
            z1 = min(nz, z0 + zb)
            blob = field[:, :, z0:z1].transpose(2, 1, 0).tobytes()
            ds.SetDataSubVolumeAs1DArrayBytes(blob, 0, 0, z0, 0, 0, nx, ny, z1 - z0)
            calls += 1
            prog.band()
    else:
        yb = max(1, budget // max(1, nx))           # slice too big: tile in Y-strips
        for z in range(nz):
            for y0 in range(0, ny, yb):
                y1 = min(ny, y0 + yb)
                blob = field[:, y0:y1, z:z + 1].transpose(2, 1, 0).tobytes()
                ds.SetDataSubVolumeAs1DArrayBytes(blob, 0, y0, z, 0, 0, nx, y1 - y0, 1)
                calls += 1
            prog.band()
    return calls


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
# File selection
# -----------------------------------------------------------------------

def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_config(cfg: dict) -> None:
    try:
        with open(_CONFIG_PATH, "w") as f:
            json.dump(cfg, f)
    except Exception:
        pass


def _resolve_syk_path(ims_dir: str, ims_stem: str) -> str | None:
    """
    Find the .syk for the open .ims.

    Order: same-dir/same-stem, then case-insensitive scan of the .ims directory,
    then a GUI file picker (the .syk may live in another folder or be named
    differently from the .ims).  Returns the path, or None if nothing was chosen.
    """
    cand = os.path.join(ims_dir, ims_stem + ".syk")
    if os.path.exists(cand):
        return cand

    target = (ims_stem + ".syk").lower()
    try:
        for fname in os.listdir(ims_dir):
            if fname.lower() == target:
                return os.path.join(ims_dir, fname)
    except Exception:
        pass

    _log(f"[syglass to imaris] no .syk next to the .ims (looked for {cand}); opening file picker.")
    return _ask_for_syk(ims_dir)


def _ask_for_syk(initialdir: str) -> str | None:
    """Open a modal file dialog for the .syk; remember its directory for next time."""
    cfg = _load_config()
    start = initialdir if os.path.isdir(initialdir) else cfg.get("last_syk_dir", "")
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select the syGlass .syk mask file",
            initialdir=start or None,
            filetypes=[("syGlass mask", "*.syk"), ("All files", "*.*")],
        )
        root.destroy()
        if path:
            cfg["last_syk_dir"] = os.path.dirname(path)
            _save_config(cfg)
            return path
        _log("[syglass to imaris] file picker cancelled.")
    except Exception as exc:
        _log(f"[syglass to imaris] file dialog unavailable: {exc}")
    return None


# -----------------------------------------------------------------------
# Progress reporting
# -----------------------------------------------------------------------

class _Progress:
    """
    Progress window (tkinter) that shows the current PHASE and an estimated wall-clock
    finish TIME.

    Two kinds of update:
      • phase(text) — the fixed one-off stages (reading mask, inventorying labels) that run
        before any per-label work.  No label count needed; the bar sits at its current value.
      • the per-label cycle: preparing() → band()×N (upload) → detecting() → end_label().

    ETA is *live*: during a label's upload the current label's duration is projected from how
    far its band ticks have progressed (elapsed / fraction_done), so even a single-label run
    — the common case — gets a moving finish time instead of a useless "estimating…".  Labels
    already completed contribute their measured mean for the ones still to come.

    The bar is always determinate and fills strictly left→right; it is never switched to
    indeterminate mode (AddSurface freezes the Python thread, so an animated bar would just
    stall mid-bounce).  band() ticks repaint the GUI only (throttled) — never logged.
    """

    _SCALE = 1000   # bar maximum (permille) — decouples the bar from label/tick counts

    def __init__(self, title: str) -> None:
        self.start = time.time()
        self.n_labels = None           # unknown until start_labels()
        self.times: list[float] = []   # durations of completed labels
        self.cur = 0                   # current label index (0-based)
        self.cur_tick = 0
        self.cur_ticks = 1             # upload ticks declared for the current label
        self.every = 1                 # GUI-repaint throttle for the current label
        self.label_start = None        # wall time the current label's UPLOAD began
        self.ok = False
        self.root = None
        self.bar = None
        self.caption = None
        try:
            import tkinter as tk
            from tkinter import ttk
            self.root = tk.Tk()
            self.root.title(title)
            self.root.attributes("-topmost", True)
            self.caption = tk.Label(self.root, text="Starting…", width=76, anchor="w")
            self.caption.pack(padx=12, pady=(12, 4))
            self.bar = ttk.Progressbar(self.root, length=480, mode="determinate",
                                       maximum=self._SCALE)
            self.bar.pack(padx=12, pady=(0, 12))
            self.ok = True
            self.root.update()
        except Exception as exc:
            _log(f"[syglass to imaris] progress GUI unavailable ({exc}); using console output.")

    @staticmethod
    def _fmt(secs: float) -> str:
        m, s = divmod(int(max(0, secs)), 60)
        return f"{m}:{s:02d}"

    def _finish(self) -> str:
        """Estimated wall-clock finish, projecting the current label from its band progress."""
        now = time.time()
        per_label = (sum(self.times) / len(self.times)) if self.times else None
        if self.label_start is not None and self.cur_tick > 0:
            frac = min(1.0, self.cur_tick / self.cur_ticks)
            cur_elapsed = now - self.label_start
            remaining_cur = max(0.0, cur_elapsed / frac - cur_elapsed) if frac > 0 else None
        elif per_label is not None:
            remaining_cur = per_label            # not uploading yet; fall back to the mean
        else:
            remaining_cur = None
        if remaining_cur is None:
            return "estimating…"
        labels_left = (self.n_labels - self.cur - 1) if self.n_labels else 0
        est_rest = (per_label if per_label is not None else remaining_cur) * max(0, labels_left)
        finish = datetime.datetime.fromtimestamp(now + remaining_cur + est_rest)
        return "~" + finish.strftime("%H:%M:%S")

    def _value(self) -> int:
        if not self.n_labels:
            return 0
        intra = min(self.cur_tick / self.cur_ticks, 1.0) if self.cur_ticks else 1.0
        frac = min(1.0, (self.cur + intra) / self.n_labels)
        return int(frac * self._SCALE)

    def _refresh(self, caption: str) -> None:
        if not self.ok:
            return
        try:
            self.bar.config(value=self._value())
            self.caption.config(text=caption)
            self.root.update()
        except Exception:
            self.ok = False

    def _label_caption(self, phase: str) -> None:
        self._refresh(f"Label {self.cur + 1}/{self.n_labels} — {phase} — "
                      f"elapsed {self._fmt(time.time() - self.start)} — finish {self._finish()}")

    def phase(self, text: str) -> None:
        """A one-off setup stage before per-label work (e.g. reading, inventorying)."""
        self._refresh(f"{text} — elapsed {self._fmt(time.time() - self.start)}")

    def start_labels(self, n_labels: int) -> None:
        self.n_labels = max(1, n_labels)

    def preparing(self, idx: int) -> None:
        """Called before the (blocking) prep result is awaited, so the caption shows why."""
        self.cur = idx
        self.cur_tick = 0
        self.cur_ticks = 1
        self.label_start = None
        self._label_caption("preparing (smoothing mask)…")

    def begin_label(self, idx: int, ticks: int) -> None:
        self.cur = idx
        self.cur_tick = 0
        self.cur_ticks = max(1, ticks)
        self.every = max(1, self.cur_ticks // 50)   # repaint at most ~50 times per label
        self.label_start = None
        self._label_caption("uploading")

    def band(self) -> None:
        """Advance one upload tick; repaint the GUI only (throttled), never logs."""
        if self.label_start is None:
            self.label_start = time.time()          # first band = upload start (for ETA)
        self.cur_tick += 1
        if self.cur_tick % self.every == 0 or self.cur_tick >= self.cur_ticks:
            self._label_caption("uploading")

    def detecting(self) -> None:
        """Caption only; leave the determinate bar in place (see class note)."""
        self.cur_tick = self.cur_ticks
        self._label_caption("meshing surface (Imaris working)…")

    def end_label(self, duration: float) -> None:
        self.times.append(duration)
        self.cur_tick = self.cur_ticks
        self._label_caption("done")

    def close(self) -> None:
        if self.ok:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.ok = False


# -----------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------
def _log_warning(vApp, message: str) -> None:
    """Show a warning dialog and print to stdout."""
    _log(f"[syglass to imaris] WARNING: {message}")
    try:
        import tkinter.messagebox
        tkinter.messagebox.showwarning("Pulse", message)
    except Exception:
        pass
