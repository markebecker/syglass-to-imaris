"""
Imaris XTension: Import syGlass Masks and Counting Points into Imaris.

Runs inside a live Imaris session.  Imaris calls XTImportFromSyGlass(imarisFile)
where imarisFile is the numeric handle for the ImarisLib COM connection.

Usage
-----
1. Open the .ims file in Imaris.
2. Run Extensions → Import from syGlass.  The matching .syk is found automatically
   if it sits beside the .ims; otherwise a file picker opens so it can be anywhere.

Strategy
--------
Masks:
  1. Parse the .syk octree directly, rather than using the `syglass` Python API (which
     has proven to be more trouble than it seems like it would be).  Block headers are
     scanned to inventory the octree; the root block (a whole-volume preview downsampled
     by 2**max_level) gives a cheap bounding-box estimate; only the deepest-level blocks
     inside that box are then read and composited.  Blocks are stored (CZ, CY, CX) and
     overlap their neighbours — see _read_mask_from_syk for the per-axis geometry.
  2. For each label, build a signed uint8 field cropped to that label's bounding box
     (sigma <= 0: binary 100=inside / 200=outside; sigma > 0: a clipped Gaussian
     gradient), and upload it in bands via SetDataSubVolumeAs1DArrayBytes, each call
     kept under the Ice message limit.
  3. Then call ISurfaces.AddSurface once — Imaris meshes the zero crossing for all of
     that label's blobs.  Prep for the next label runs on a worker thread while the
     current one uploads.

Progress + logs:
  A tkinter window shows the estimated finish time.  A timestamped log is written to a
  'logs' folder inside the script's own directory (…/<script_dir>/logs/import_from_syglass_xt_*.log).

Spots (counting points):
  Read countingPoints from .sym LevelDB and create ISpots.  EXPERIMENTAL and off by
  default — the on-disk record layout and the syGlass → Imaris coordinate transform are
  both unconfirmed, so imported positions should not be trusted.  Enable it from the
  options menu only to help work the format out.  I have never tested this so it likely
  will not work at all.

Limitations:
  Single timepoint. Multichannel is fine. Points/spots likely nonfunctional.

Troubleshooting:
    - Ensure that the directory containing this script is in the Imaris Preferences CustomTools → Python path list
    - Check the log file for errors.
    - Turn on diagnostics in the options menu to see ASCII max-projections of the reconstructed mask (shape sanity check, independent of Imaris).
    - Check in Task Manager- Python and Imaris should use resources.

Python 3.11 compatibility required (Imaris bundled interpreter).
"""

from __future__ import annotations

import datetime
import json
import os
import posixpath
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
import ImarisLib  # type: ignore  # noqa: E402
# NOTE: the syglass/pyglass API path was removed.  pyglass seems to be geared toward .syg-based
# projects and did not work well with the .syk/.ims pairs used here, so masks come straight
# from the .syk parser (_read_mask_from_syk).


# -----------------------------------------------------------------------
# Tunables
# -----------------------------------------------------------------------

# Largest payload (bytes) to push in a single COM call.  Ice's default MessageSizeMax
# is ~1 MB; staying under it lets us upload a band of z-slices per call.
_ICE_CHUNK_BYTES = 900_000

# syGlass label IDs are generally small.  Anything above this is probably storage artifact rather than a
# painted label — see the padding-column note in _read_mask_from_syk — and is erased at
# read time so it cannot punch holes in a neighbouring label's surface.
_MAX_PLAUSIBLE_LABEL = 100

# Extra root-level blocks of margin on each side of the estimated mask bounding box.
_ROOT_BBOX_MARGIN_BLOCKS = 2

# Minimum crop padding (voxels) around a label's bounding box, so a blurred field has
# room to fall back to "outside" without touching the crop wall.
_PREP_MARGIN_VOXELS = 4

# Radius given to imported counting points.
_SPOT_RADIUS_UM = 0.5

# Warn when the reconstructed syGlass grid and the .ims voxel grid disagree by more than
# this fraction — the mask is stretched to the .ims extents, so a mismatch is problematic.
_GRID_RATIO_TOLERANCE = 0.01

# Block-overlap self-check (see _measure_block_aprons): how many adjacent block pairs per
# axis to sample on a normal run, how much a shared apron must beat the baseline
# agreement to count as present, and how many raw blocks the reader may cache.
_APRON_CHECK_PAIRS = 8
_APRON_MARGIN = 0.30
_APRON_CACHE_BLOCKS = 64

# Remembers the options-menu choices between runs.
_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".import_from_syglass_xt.json")

# Open log file handle (set by _setup_logging); every _log() line carries a timestamp
# so it is clear which step consumes time.  None until setup, in which case _log()
# still prints to the console.
_LOG_FH = None


# -----------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------

def _setup_logging():
    """
    Open a timestamped log file in a 'logs' folder inside the script's own directory —
    i.e. <script_dir>/logs/import_from_syglass_xt_<timestamp>.log.  Returns the path, or None on failure.
    """
    global _LOG_FH
    if _LOG_FH is not None:            # a previous run in this Imaris session left one open
        try:
            _LOG_FH.close()
        except Exception:
            pass
        _LOG_FH = None
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir  = os.path.join(script_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        stamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"import_from_syglass_xt_{stamp}.log")
        _LOG_FH  = open(log_path, "w", buffering=1)  # line-buffered
        return log_path
    except Exception as exc:
        _LOG_FH = None
        _log(f"could not open log file ({exc}); console only.")
        return None


def _log(msg: str) -> None:
    """Print a timestamped, prefixed line to the console and, if open, the log file."""
    line = f"{datetime.datetime.now().strftime('%H:%M:%S')} [syglass to imaris] {msg}"
    print(line)
    if _LOG_FH is not None:
        try:
            _LOG_FH.write(line + "\n")
        except Exception:
            pass


def _warn(msg: str) -> None:
    """Log a line that the user should notice when reading back the log."""
    _log(f"WARNING: {msg}")


# -----------------------------------------------------------------------
# XTension entry point (Imaris calls this)
# -----------------------------------------------------------------------

#   <CustomTools>
#       <Menu>
#           <Item name="Import from syGlass (Development)" icon="Python3"
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
        # Try both paths — at least one should be writable
        for log_path in [
            os.path.join(os.path.expanduser("~"), "import_from_syglass_xt_crash.log"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "import_from_syglass_xt_crash.log"),
        ]:
            try:
                with open(log_path, "a") as f:
                    f.write(tb)
                _log(f"CRASH — traceback written to: {log_path}")
                break
            except Exception:
                pass
        print(tb)  # also try stdout in case console is still open


def _run(imarisFile: int) -> None:
    """Inner implementation (wrapped for crash logging)."""
    log_path = _setup_logging()
    if log_path:
        _log(f"logging to: {log_path}")

    vLib = ImarisLib.ImarisLib()
    vApp = vLib.GetApplication(imarisFile)

    # ----------------------------------------------------------------
    # 1. Locate companion syGlass files
    # ----------------------------------------------------------------
    ims_path = vApp.GetCurrentFileName()
    if not ims_path:
        _log_warning("No file is currently open in Imaris.")
        return

    ims_dir  = os.path.dirname(ims_path)
    ims_stem = os.path.splitext(os.path.basename(ims_path))[0]

    # Locate the .syk: try the conventional same-dir/same-stem path first, then a
    # case-insensitive scan, then fall back to a GUI file picker (the .syk may live
    # elsewhere or be named differently from the .ims).
    syk_path = _resolve_syk_path(ims_dir, ims_stem)
    if not syk_path or not os.path.exists(syk_path):
        _log_warning("No .syk file selected — aborting import.")
        return

    # The .sym (counting points / metadata) sits next to the chosen .syk, sharing its stem.
    syk_dir  = os.path.dirname(syk_path)
    syk_stem = os.path.splitext(os.path.basename(syk_path))[0]
    sym_path = os.path.join(syk_dir, syk_stem + ".sym")
    _log(f"using .syk: {syk_path}")

    settings = _ask_settings()
    if settings is None:
        _log("options menu cancelled — nothing imported.")
        return
    _log(f"options: smoothing sigma={settings['sigma']}  "
         f"diagnostics={settings['diagnostics']}  spots={settings['spots']}")

    # ----------------------------------------------------------------
    # 2. Read image dimensions and physical extents from Imaris
    # ----------------------------------------------------------------
    geom = _Geometry(vApp.GetDataSet())
    _log("Importing your syglass mask...")
    for line in geom.describe():
        _log(line)
    if geom.size_t > 1:
        _warn(f".ims has {geom.size_t} timepoints; surfaces are built from a single volume "
              f"and will land on t=0 only.")

    # ----------------------------------------------------------------
    # 3. Read syGlass label mask
    # ----------------------------------------------------------------
    # Progress window is created here (before the multi-second read) so the user sees each
    # phase; it is closed in the finally at the end of the surface loop.
    prog = _Progress("syGlass → Imaris — importing mask")

    prog.phase("Reading syGlass mask from .syk")
    t_read0 = time.time()
    # label_vol: (NX, NY, NZ) uint16, 0=background, 1..N=label.
    # clip_info: (vx0, vy0, vz0, full_nx, full_ny, full_nz) — the sub-region that was read.
    label_vol, clip_info = _read_mask_from_syk(syk_path, diagnostics=settings["diagnostics"])

    if label_vol is None or not np.any(label_vol):
        prog.close()
        _log_warning("No mask data found in .syk file (or file is empty).")
        return
    _log(f"mask read in {time.time() - t_read0:.1f}s")

    prog.phase("Inventorying labels")
    t_scan0 = time.time()
    label_ids, counts = _inventory_labels(label_vol)
    _log(f"label_vol shape: {label_vol.shape}, dtype: {label_vol.dtype}")
    _log(f"Non-zero voxels: {int(label_vol.size - counts[0])} / {label_vol.size}  "
         f"(inventory in {time.time() - t_scan0:.1f}s)")
    for v in label_ids:
        _log(f"  label {v}: {int(counts[v])} voxels")
    _log(f"Found {len(label_ids)} labels: {label_ids}")
    if not label_ids:
        prog.close()
        _log_warning("The .syk contains voxel data but no usable label IDs — nothing to mesh.")
        return
    prog.start_labels(len(label_ids))

    # Troubleshooting only: ASCII max-projections of the reconstructed mask (shape sanity
    # check, independent of Imaris).
    if settings["diagnostics"]:
        prog.phase("Rendering mask preview")
        _log_mask_projections(label_vol)

    # ----------------------------------------------------------------
    # 4. Build one ISurfaces per label
    # ----------------------------------------------------------------
    factory = vApp.GetFactory()
    scene   = vApp.GetSurpassScene()
    try:
        _build_surfaces(factory, scene, label_vol, label_ids, geom, clip_info,
                        settings["sigma"], prog, ims_stem)
    finally:
        prog.close()

    # ----------------------------------------------------------------
    # 5. Import counting points (spots) from .sym — experimental, opt-in
    # ----------------------------------------------------------------
    if settings["spots"] and os.path.exists(sym_path):
        _import_counting_points(factory, scene, sym_path, geom, ims_stem)

    # ----------------------------------------------------------------
    # 6. Done — intentionally NO save.
    #    vApp.FileSave("", "") pops Imaris's Save dialog (empty filename) and rewrites the
    #    whole .ims (~3 min on large files).  The surfaces are already in the live Surpass
    #    scene, so we leave saving to the user (Ctrl+S) whenever they choose.
    # ----------------------------------------------------------------
    _log("Done — surfaces added to the scene. NOT saved: press Ctrl+S "
         "in Imaris to write them to the .ims when ready.")


# -----------------------------------------------------------------------
# Dataset geometry
# -----------------------------------------------------------------------

class _Geometry:
    """Voxel dimensions and physical extents (µm) of the .ims dataset open in Imaris."""

    def __init__(self, ds) -> None:
        self.size = np.array([ds.GetSizeX(), ds.GetSizeY(), ds.GetSizeZ()], dtype=np.int64)
        try:
            self.size_t = int(ds.GetSizeT())
        except Exception:
            self.size_t = 1
        self.ext_min = np.array([ds.GetExtendMinX(), ds.GetExtendMinY(), ds.GetExtendMinZ()])
        self.ext_max = np.array([ds.GetExtendMaxX(), ds.GetExtendMaxY(), ds.GetExtendMaxZ()])
        self.voxel_size = (self.ext_max - self.ext_min) / self.size

    def describe(self) -> list[str]:
        nx, ny, nz = self.size
        return [
            f".ims dims:    {nx} x {ny} x {nz} voxels",
            f".ims extents: X=[{self.ext_min[0]:.3f}, {self.ext_max[0]:.3f}] "
            f"Y=[{self.ext_min[1]:.3f}, {self.ext_max[1]:.3f}] "
            f"Z=[{self.ext_min[2]:.3f}, {self.ext_max[2]:.3f}]",
            f".ims voxel size (um): {self.voxel_size[0]:.4f} x {self.voxel_size[1]:.4f} "
            f"x {self.voxel_size[2]:.4f}",
        ]


def _clip_extents(geom: _Geometry, clip_info, vol_shape):
    """
    Physical extents (µm) spanned by the reconstructed volume.

    The mask is stretched onto the .ims extents, so the µm size of one syGlass voxel comes
    from the FULL syGlass grid; the clipped sub-volume then occupies its own slice of that.
    Returns (lo_xyz, hi_xyz, full_grid_xyz).
    """
    shape = np.asarray(vol_shape, dtype=np.float64)
    if clip_info is None:
        return geom.ext_min.copy(), geom.ext_max.copy(), shape
    origin = np.asarray(clip_info[:3], dtype=np.float64)
    full   = np.asarray(clip_info[3:], dtype=np.float64)
    per_voxel = (geom.ext_max - geom.ext_min) / full
    lo = geom.ext_min + origin * per_voxel
    hi = geom.ext_min + (origin + shape) * per_voxel
    return lo, hi, full


def _inventory_labels(label_vol: np.ndarray):
    """
    Count voxels per label ID in one pass.

    np.unique would sort the whole (billions of voxels) array; a histogram is a single O(n)
    scan whose bins index the label value directly.  It is accumulated over slabs because
    np.bincount casts its input to intp — on a 4-billion-voxel array that would be a 32 GB
    temporary for no benefit.

    Returns (label_ids, counts) where counts is indexed by label value.
    """
    counts = np.zeros(2 ** 16, dtype=np.int64)
    slab = max(1, 64)
    for x0 in range(0, label_vol.shape[0], slab):
        block = label_vol[x0:x0 + slab]
        counts += np.bincount(block.ravel(), minlength=2 ** 16)

    ids = [int(v) for v in np.nonzero(counts)[0] if v != 0]
    # Implausible IDs are erased while blocks are placed (see _read_mask_from_syk), so this
    # should never fire; it stays as a net in case a file uses a geometry we haven't seen.
    implausible = [v for v in ids if v > _MAX_PLAUSIBLE_LABEL]
    if implausible:
        _warn(f"ignoring {len(implausible)} implausible label IDs (>{_MAX_PLAUSIBLE_LABEL}): "
              f"{implausible}")
    return [v for v in ids if v <= _MAX_PLAUSIBLE_LABEL], counts


# -----------------------------------------------------------------------
# Surface construction
# -----------------------------------------------------------------------

def _resolve_uint8_type():
    """
    Resolve the eTypeUInt8 enum member used by IDataSet.Create.

    Depending on the Imaris build the Ice-generated enum hangs off ImarisLib itself or off
    a sibling 'Imaris' module.  Falls back to the raw integer 0, which is eTypeUInt8 in
    every Imaris version we have seen.
    """
    candidates = [ImarisLib, sys.modules.get("Imaris")]
    candidates += [m for name, m in sys.modules.items()
                   if m is not None and "imaris" in name.lower()]
    for mod in candidates:
        tType = getattr(mod, "tType", None) if mod is not None else None
        member = getattr(tType, "eTypeUInt8", None) if tType is not None else None
        if member is not None:
            return member
    _warn("could not resolve the tType enum; using raw integer 0 for eTypeUInt8")
    return 0


def _build_surfaces(factory, scene, label_vol, label_ids, geom: _Geometry, clip_info,
                    sigma: float, prog, name_prefix: str) -> None:
    """
    Create one ISurfaces object per label and add it to the Surpass scene.

    AddSurface interprets the dataset as a signed field and finds the surface at the zero
    crossing.  uint8 is treated as signed int8: 100 → +100 (inside), 200 → −56 (outside).
    Each label's IDataSet is sized to that label's OWN bounding box (not the full
    reconstruction — which may be billions of mostly-empty voxels), with physical extents
    for just that sub-region.  Data is uploaded in chunks kept under the Ice message limit.

    Prefetch depth 1: prep for the NEXT label (crop + smooth + signed-field build) runs on a
    background thread while the main thread uploads the CURRENT one, so the pure-NumPy prep
    overlaps the serial COM I/O.
    """
    eTypeUInt8 = _resolve_uint8_type()
    vol_shape = np.array(label_vol.shape)
    ds_lo, ds_hi, full_grid = _clip_extents(geom, clip_info, vol_shape)

    if clip_info is not None:
        _log(f"IDataSet (clipped): {vol_shape[0]}×{vol_shape[1]}×{vol_shape[2]} voxels  "
             f"full grid {int(full_grid[0])}×{int(full_grid[1])}×{int(full_grid[2])}")
    _log(f"IDataSet extents: X=[{ds_lo[0]:.1f}, {ds_hi[0]:.1f}] "
         f"Y=[{ds_lo[1]:.1f}, {ds_hi[1]:.1f}] Z=[{ds_lo[2]:.1f}, {ds_hi[2]:.1f}]")

    # The mask is stretched to fill the .ims extents, so a syGlass grid that does not match
    # the .ims voxel grid is a registration error rather than a cosmetic mismatch.
    ratios = geom.size / full_grid
    _log(f"ims/syk ratio: X={ratios[0]:.2f}  Y={ratios[1]:.2f}  Z={ratios[2]:.2f}")
    for name, ratio in zip("XYZ", ratios):
        if abs(ratio - 1.0) > _GRID_RATIO_TOLERANCE:
            _warn(f"{name}: reconstructed syGlass grid is {abs(1.0 - ratio) * 100:.1f}% "
                  f"{'smaller' if ratio > 1 else 'larger'} than the .ims grid. The mask is "
                  f"scaled onto the .ims extents, so surfaces may be misregistered along "
                  f"{name}. Check the block geometry in _read_mask_from_syk against this file.")

    # Physical size of one clip voxel — converts each label's crop (in clip-voxel coords)
    # into the µm extents of its own IDataSet.
    per_voxel = (ds_hi - ds_lo) / vol_shape

    n_labels = len(label_ids)
    _log(f"{n_labels} label(s); each cropped to its bbox, "
         f"uploaded in <= {_ICE_CHUNK_BYTES // 1000} KB chunks; prep pipelined 1 ahead")

    t_run0 = time.time()
    sum_prep = sum_upload = sum_detect = 0.0

    # One worker is enough: the loop only submits the next prep after collecting the
    # current one, so at most one prep is ever in flight.
    with ThreadPoolExecutor(max_workers=1) as pool:
        next_future = pool.submit(_prep_label, label_vol, label_ids[0], sigma)
        for label_idx, label_id in enumerate(label_ids):
            tag = f"L{label_idx + 1}/{n_labels} (id {label_id})"

            prog.preparing(label_idx)      # caption shown during the (blocking) prep wait
            t0 = time.time()
            prep = next_future.result()
            t_prep = time.time() - t0
            # Kick off prep for the next label so it overlaps this label's upload.
            if label_idx + 1 < n_labels:
                next_future = pool.submit(_prep_label, label_vol, label_ids[label_idx + 1], sigma)

            if prep is None:
                _log(f"{tag}: empty — skipped")
                prog.begin_label(label_idx, 1)
                prog.end_label(0.0)
                continue

            field, offset, n_painted, n_kept = prep
            crop = np.array(field.shape)
            prog.begin_label(label_idx, _upload_ticks(*crop))

            # Physical extents of this label's crop (sub-region of the clip volume).
            lo = ds_lo + np.asarray(offset) * per_voxel
            hi = ds_lo + (np.asarray(offset) + crop) * per_voxel
            _log(f"{tag}: crop {crop[0]}x{crop[1]}x{crop[2]} "
                 f"({crop.prod() / 1e6:.1f} MB) at voxel "
                 f"({offset[0]},{offset[1]},{offset[2]}); painted={n_painted} kept={n_kept}; "
                 f"bbox um X=[{lo[0]:.0f},{hi[0]:.0f}] Y=[{lo[1]:.0f},{hi[1]:.0f}] "
                 f"Z=[{lo[2]:.0f},{hi[2]:.0f}]")

            sdf_ds = factory.CreateDataSet()
            sdf_ds.Create(eTypeUInt8, int(crop[0]), int(crop[1]), int(crop[2]), 1, 1)
            sdf_ds.SetExtendMinX(lo[0]); sdf_ds.SetExtendMinY(lo[1]); sdf_ds.SetExtendMinZ(lo[2])
            sdf_ds.SetExtendMaxX(hi[0]); sdf_ds.SetExtendMaxY(hi[1]); sdf_ds.SetExtendMaxZ(hi[2])

            t0 = time.time()
            _upload_field(sdf_ds, field, prog)
            t_up = time.time() - t0
            del field, prep   # free this label's prep before the next is awaited

            # ONE AddSurface on the whole cropped field — meshes all of the label's blobs
            # at once.
            surfaces = factory.CreateSurfaces()
            prog.detecting()
            t0 = time.time()
            surfaces.AddSurface(sdf_ds, 0)
            t_det = time.time() - t0
            try:
                n_surf = surfaces.GetNumberOfSurfaces()
            except Exception as exc:
                n_surf = f"?({exc})"

            r, g, b = _DEFAULT_PALETTE[(label_id - 1) % len(_DEFAULT_PALETTE)]
            surfaces.SetColorRGBA(_pack_rgba(r, g, b, 255))
            surfaces.SetName(f"{name_prefix} label {label_id}")
            scene.AddChild(surfaces, -1)

            sum_prep += t_prep; sum_upload += t_up; sum_detect += t_det
            _log(f"{tag}: prep_wait={t_prep:.2f}s  upload={t_up:.2f}s  "
                 f"detect={t_det:.2f}s  surfaces_in_object={n_surf}")
            prog.end_label(t_prep + t_up + t_det)

    t_total = time.time() - t_run0
    _log(f"Created {n_labels} surface object(s) in {t_total:.1f}s  "
         f"(prep_wait {sum_prep:.1f}s + upload {sum_upload:.1f}s + detect {sum_detect:.1f}s; "
         f"avg {t_total / max(1, n_labels):.2f}s/label)")


def _import_counting_points(factory, scene, sym_path: str, geom: _Geometry,
                            name_prefix: str) -> None:
    """Read counting points from the .sym and add them to the scene as ISpots."""
    spots_xyz = _read_counting_points_from_sym(sym_path, geom)
    if spots_xyz is None or len(spots_xyz) == 0:
        return
    n = len(spots_xyz)
    vSpots = factory.CreateSpots()
    vSpots.Set(spots_xyz.tolist(), [0] * n, [_SPOT_RADIUS_UM] * n)
    vSpots.SetName(name_prefix + " (syGlass counting points — EXPERIMENTAL)")
    scene.AddChild(vSpots, -1)
    _log(f"Imported {n} counting points.")


# -----------------------------------------------------------------------
# Mask reader: manual .syk parser (main)
# -----------------------------------------------------------------------

def _read_mask_from_syk(syk_path: str, diagnostics: bool = False):
    """
    Read the syGlass label mask from a .syk file.

    Scans all block headers to inventory the octree, reads the root block to
    estimate the mask bounding box, then composites only the deepest-level
    blocks that fall within that bbox.  This avoids allocating the full
    (potentially huge) reconstructed volume.

    Returns ((NX, NY, NZ) uint16 array, clip_info) where
      clip_info = (vx0, vy0, vz0, full_nx, full_ny, full_nz)
    or (None, None) on failure.
    """
    FVGU = b"fvgu"
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

            _log(f".syk: cx={cx} cy={cy} cz={cz}  "
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
                _log(".syk: no valid blocks found")
                return None, None

            max_level = max(_syk_block_level(bid) for bid in block_offset)
            n_grid    = 2 ** max_level
            # Per-axis block geometry (measured from real .syk).  X (the fastest-stored
            # axis) carries a 1-voxel apron PLUS 2 trailing PADDING columns that copy the
            # block's own column 0 (the 18024/30825 sentinels are these).  Placing them
            # injects a ghost plane one block-width away in X — so drop X's apron + 2
            # padding (last 3).  Y and Z carry only a 1-voxel apron (drop last 1).
            # ax/ay/az = unique voxels kept per block per axis (= placement size = stride).
            # _check_block_aprons below re-measures this on the file at hand and warns if
            # the assumption does not hold.
            ax, ay, az = cx - 3, cy - 1, cz - 1
            full_nx, full_ny, full_nz = n_grid * ax, n_grid * ay, n_grid * az

            _log(f".syk: {len(block_offset)} blocks; "
                 f"deepest level={max_level}  full grid={full_nx}×{full_ny}×{full_nz}")

            _check_block_aprons(f, block_offset, cx, cy, cz, max_level, block_payload,
                                verbose=diagnostics)

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
                    m = _ROOT_BBOX_MARGIN_BLOCKS
                    bbox_x0 = max(0,       (int(nz_r[0].min()) - m) * n_grid)
                    bbox_x1 = min(full_nx, (int(nz_r[0].max()) + m + 1) * n_grid)
                    bbox_y0 = max(0,       (int(nz_r[1].min()) - m) * n_grid)
                    bbox_y1 = min(full_ny, (int(nz_r[1].max()) + m + 1) * n_grid)
                    bbox_z0 = max(0,       (int(nz_r[2].min()) - m) * n_grid)
                    bbox_z1 = min(full_nz, (int(nz_r[2].max()) + m + 1) * n_grid)

        clip_nx = bbox_x1 - bbox_x0
        clip_ny = bbox_y1 - bbox_y0
        clip_nz = bbox_z1 - bbox_z0
        _log(f".syk: allocating clipped vol {clip_nx}×{clip_ny}×{clip_nz} "
             f"(origin {bbox_x0},{bbox_y0},{bbox_z0})")

        # uint16 (not uint32): the .syk payload is already uint16 and label IDs fit, so
        # this halves the footprint of the volume and the bandwidth of every scan over it.
        vol = np.zeros((clip_nx, clip_ny, clip_nz), dtype=np.uint16)

        # Pass 2 — place every LEAF block within the bbox.
        #
        # The octree is adaptive: a block whose children are absent from the file is a leaf
        # holding its region's mask at its own (coarser) resolution.  Reading only the
        # deepest level therefore drops data — on one test file 478 of 512 level-3 blocks
        # were childless and carried mask, so all but 34 regions came out empty.  Whether a
        # given file is adaptive or uniform varies: a second file stored everything at the
        # deepest level and read correctly either way.  tools/octree_structure.py reports
        # which shape a file has.
        #
        # A leaf at level L is coarser than the finest grid by scale = 2**(max_level - L),
        # so each of its voxels covers a scale**3 cube and is expanded on placement.
        has_child = set()
        for bid in block_offset:
            if bid:
                has_child.add((bid - 1) // 8)

        with open(syk_path, "rb") as f:
            n_placed = 0
            n_sentinel = 0
            per_level = {}
            for bid, boff in block_offset.items():
                if bid in has_child:
                    continue                      # interior node: its children hold the data
                lv, ix, iy, iz = _syk_block_position(bid)
                scale = 1 << (max_level - lv)     # finest voxels per voxel of this block
                sx, sy, sz = ax * scale, ay * scale, az * scale   # span in finest voxels

                bx0, bx1 = ix * sx, ix * sx + sx
                by0, by1 = iy * sy, iy * sy + sy
                bz0, bz1 = iz * sz, iz * sz + sz

                if bx1 <= bbox_x0 or bx0 >= bbox_x1: continue
                if by1 <= bbox_y0 or by0 >= bbox_y1: continue
                if bz1 <= bbox_z0 or bz0 >= bbox_z1: continue

                f.seek(boff + 24)
                raw = f.read(block_payload)
                # reshape is (cz, cy, cx).  Drop last z & last y (aprons) and the last 3 x
                # (X apron + 2 padding columns) → keep the unique (cx-3, cy-1, cz-1) voxels.
                arr = (np.frombuffer(raw, dtype="<u2")
                       .reshape(cz, cy, cx)[:-1, :-1, :-3]
                       .transpose(2, 1, 0))   # (cx-3, cy-1, cz-1), uint16
                if scale > 1:
                    arr = arr.repeat(scale, 0).repeat(scale, 1).repeat(scale, 2)

                cx0 = max(0, bbox_x0 - bx0);  cx1 = min(sx, bbox_x1 - bx0)
                cy0 = max(0, bbox_y0 - by0);  cy1 = min(sy, bbox_y1 - by0)
                cz0 = max(0, bbox_z0 - bz0);  cz1 = min(sz, bbox_z1 - bz0)

                dx0 = bx0 + cx0 - bbox_x0;  dx1 = dx0 + (cx1 - cx0)
                dy0 = by0 + cy0 - bbox_y0;  dy1 = dy0 + (cy1 - cy0)
                dz0 = bz0 + cz0 - bbox_z0;  dz1 = dz0 + (cz1 - cz0)

                dst = vol[dx0:dx1, dy0:dy1, dz0:dz1]     # view into vol
                dst[...] = arr[cx0:cx1, cy0:cy1, cz0:cz1]

                # Erase storage sentinels (18024/30825 and friends) as each block lands.
                # With the strides above they are sliced off before placement, so this
                # should find nothing; it costs one pass over a 6 MB block and keeps a
                # phantom "label 18024" out of the inventory if a file ever slips one
                # through.  A sentinel reads as "not this label" whether it is erased or
                # left in place, so this neither creates nor closes a hole.
                bad = dst > _MAX_PLAUSIBLE_LABEL
                n_bad = int(bad.sum())
                if n_bad:
                    dst[bad] = 0
                    n_sentinel += n_bad
                n_placed += 1
                per_level[lv] = per_level.get(lv, 0) + 1

        levels_txt = ", ".join(f"L{lv}:{n}" for lv, n in sorted(per_level.items()))
        _log(f".syk: placed {n_placed} leaf block(s) — {levels_txt}"
             f"  (deepest level {max_level}, {n_grid ** 3} slots)")
        if any(lv < max_level for lv in per_level):
            coarse = sum(n for lv, n in per_level.items() if lv < max_level)
            _log(f".syk: adaptive octree — {coarse} leaf block(s) sit above the deepest "
                 f"level and were upsampled into place")
        if n_sentinel:
            _log(f".syk: erased {n_sentinel} sentinel voxel(s) with IDs > "
                 f"{_MAX_PLAUSIBLE_LABEL} before meshing")

        # No seam-filling pass: with correct per-axis strides the blocks join exactly, and
        # a whole-volume repair scan cost ~20s and tens of GB to patch nothing.  If a file
        # ever does show 1-voxel gaps at block boundaries, the strides are wrong for it —
        # _check_block_aprons above will have warned, and the seam probe below shows the
        # damage directly.  Fix the geometry rather than papering over it here.
        #
        # Troubleshooting only: value profile across block boundaries (should read '1111|1111').
        if diagnostics:
            _probe_seams(vol, ax, ay, az, (bbox_x0, bbox_y0, bbox_z0))

        clip_info = (bbox_x0, bbox_y0, bbox_z0, full_nx, full_ny, full_nz)
        return vol, clip_info

    except PermissionError as exc:
        _log(f".syk parse failed: {exc}")
        _log("NOTE: the .syk is locked by syGlass — close the project in syGlass and retry.")
        return None, None
    except Exception:
        _log(f".syk parse failed:\n{traceback.format_exc()}")
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
# Diagnostics (block geometry self-check, seam probe, ASCII previews)
# -----------------------------------------------------------------------

def _measure_block_aprons(f, block_offset, cx, cy, cz, max_level, block_payload,
                          max_pairs=None):
    """
    Measure how deepest-level blocks overlap their neighbours, per axis.

    For adjacent blocks A (at i) and B (at i+1) this accumulates painted-voxel agreement of:
      high = A.last-slice  vs B.first-slice   (high-side apron: A's last duplicates boundary)
      low  = A.first-slice vs prev.last-slice (low-side apron:  A's first duplicates boundary)
      base = A.mid-slice   vs B.mid-slice     (baseline for a smoothly-varying volume)
      self = A.last-slice  vs A.first-slice   (padding: A's last is a copy of its own first)
    Interpretation: high ≫ base ⇒ A's LAST slice is a shared apron; self ≈ 1 ⇒ A's last
    slice is padding rather than real data.

    Returns {"X": {...}, "Y": {...}, "Z": {...}} of ratios plus pair/voxel counts.
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
            if len(cache) >= _APRON_CACHE_BLOCKS:
                cache.clear()
            f.seek(block_offset[bid] + 24)
            cache[bid] = np.frombuffer(f.read(block_payload), dtype="<u2").reshape(cz, cy, cx)
        return cache[bid]

    def _acc(a, b, tot):
        m = (a > 0) | (b > 0)
        return (tot[0] + int(np.sum(a[m] == b[m])), tot[1] + int(m.sum()))

    def _ratio(t):
        return (t[0] / t[1]) if t[1] else float("nan")

    results = {}
    # The payload is stored (cz, cy, cx), so array axis 2 is X, 1 is Y, 0 is Z.
    for axis, name, dim in ((2, "X", cx), (1, "Y", cy), (0, "Z", cz)):
        unit = {2: (1, 0, 0), 1: (0, 1, 0), 0: (0, 0, 1)}[axis]
        hi = lo = base = slf = (0, 0)
        npairs = 0
        for pos, bid in pos2bid.items():
            if max_pairs is not None and npairs >= max_pairs:
                break
            nb = tuple(p + u for p, u in zip(pos, unit))
            if nb in pos2bid:
                A, B = _read(bid), _read(pos2bid[nb])
                hi   = _acc(np.take(A, dim - 1, axis=axis), np.take(B, 0, axis=axis), hi)
                base = _acc(np.take(A, dim // 2, axis=axis), np.take(B, dim // 2, axis=axis), base)
                slf  = _acc(np.take(A, dim - 1, axis=axis), np.take(A, 0, axis=axis), slf)
                npairs += 1
            pb = tuple(p - u for p, u in zip(pos, unit))
            if pb in pos2bid:
                A, P = _read(bid), _read(pos2bid[pb])
                lo = _acc(np.take(A, 0, axis=axis), np.take(P, dim - 1, axis=axis), lo)
        results[name] = {"high": _ratio(hi), "low": _ratio(lo), "base": _ratio(base),
                         "self": _ratio(slf), "pairs": npairs, "voxels": hi[1]}
    return results


def _check_block_aprons(f, block_offset, cx, cy, cz, max_level, block_payload,
                        verbose=False) -> None:
    """
    Verify the hardcoded per-axis block strides against the file actually being read.

    The strides in _read_mask_from_syk (X: drop 3, Y/Z: drop 1) were derived from real
    files, but they are the single most fragile assumption in this reader: get them wrong
    and every surface is misplaced.  Sampling a few block pairs costs well under a second,
    so it runs on every import and warns when the measurement disagrees.
    """
    try:
        measured = _measure_block_aprons(f, block_offset, cx, cy, cz, max_level,
                                         block_payload,
                                         max_pairs=None if verbose else _APRON_CHECK_PAIRS)
    except Exception as exc:
        _log(f"block-geometry self-check skipped ({exc})")
        return

    if verbose:
        _log("APRON DIAGNOSTIC (aggregated over adjacent painted pairs):")
        for name, m in measured.items():
            _log(f"  {name}: high(A.last=B.first)={m['high']:.2f}  "
                 f"low(A.first=prev.last)={m['low']:.2f}  base={m['base']:.2f}  "
                 f"self(A.last=A.first)={m['self']:.2f}  "
                 f"pairs={m['pairs']} vox={m['voxels']}")

    # X drops 3 columns because its trailing columns are padding, not a shared apron;
    # Y and Z drop 1 because their last slice IS the neighbour's first.
    expect_shared_apron = {"X": False, "Y": True, "Z": True}
    for name, expected in expect_shared_apron.items():
        m = measured[name]
        if not np.isfinite(m["high"]) or not np.isfinite(m["base"]):
            continue        # nothing painted near a boundary on this axis; no evidence
        observed = (m["high"] - m["base"]) > _APRON_MARGIN
        if observed and not expected:
            _warn(f"{name}: blocks appear to SHARE an apron (high={m['high']:.2f} vs "
                  f"base={m['base']:.2f}), but this reader drops 3 {name} columns as "
                  f"padding. Surfaces may be misaligned along {name}.")
        elif expected and not observed:
            _warn(f"{name}: blocks appear to ABUT (high={m['high']:.2f} vs "
                  f"base={m['base']:.2f}), but this reader drops the last {name} slice as "
                  f"a shared apron. Surfaces may be misaligned along {name}.")

    x = measured["X"]
    if np.isfinite(x["self"]) and x["self"] < 0.5 and x["voxels"] > 0:
        _warn(f"X: the trailing columns do not look like copies of column 0 "
              f"(self={x['self']:.2f}); the 3-column X drop is unverified for this file.")


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

    _log("SEAM PROBE (values across block boundaries; | = block edge):")
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
            _log(f"  {name}@{c}: {_render(prof, c - lo)}")
            shown += 1
        if shown == 0:
            _log(f"  {name}: no painted block boundary found")


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
    _log("mask projection — top-down (rows=X, cols=Y):")
    for ln in _ascii_projection(top):
        _log("  |" + ln + "|")
    _log("mask projection — side (rows=X, cols=Z):")
    for ln in _ascii_projection(side):
        _log("  |" + ln + "|")


# -----------------------------------------------------------------------
# Counting points reader from .sym
# -----------------------------------------------------------------------

def _read_counting_points_from_sym(sym_path: str, geom: _Geometry) -> np.ndarray | None:
    """
    Read syGlass counting points from .sym (ZIP-wrapped LevelDB).  EXPERIMENTAL.

    Counting points are stored under keys 'default::countingPoints::N', each a 20-byte
    record whose layout is NOT confirmed; this reads the first three float32s as XYZ.

    The syGlass → Imaris coordinate transform is also unconfirmed.  An earlier version of
    this reader added the .syk header fields at bytes 8–19 to every point as a volume
    "centre"; those fields are actually the octree BLOCK dimensions (cx, cy, cz) that
    _read_mask_from_syk uses, so that offset was simply wrong and has been removed.  What
    remains — scaling raw .syk coordinates by the .ims voxel size — is a plausible guess,
    not a verified transform.

    TODO: confirm the record layout and the coordinate frame against a file with known
    point positions, then drop the EXPERIMENTAL gate in the options menu.

    Returns (N, 3) float32 in Imaris physical µm, or None.
    """
    try:
        try:
            import leveldb  # type: ignore
        except ImportError:
            _log("leveldb not available — cannot read .sym counting points.")
            return None

        _warn("counting-point import is EXPERIMENTAL: the record layout and coordinate "
              "transform are unconfirmed, so these positions are not trustworthy.")

        tmpdir = tempfile.mkdtemp(prefix="import_from_syglass_xt_")
        try:
            with zipfile.ZipFile(sym_path, "r") as zf:
                for member in zf.namelist():
                    # Refuse absolute paths and any '..' escape before extracting.
                    norm = posixpath.normpath(member.replace("\\", "/"))
                    if norm.startswith(("/", "../")) or norm == ".." or ":" in norm.split("/")[0]:
                        _warn(f"skipping suspicious .sym entry: {member}")
                        continue
                    zf.extract(member, tmpdir)

            db = leveldb.LevelDB(tmpdir)
            points = []

            prefix = b"default::countingPoints::"
            for key, val in db.RangeIter():
                if not key.startswith(prefix):
                    continue
                if len(val) >= 12:
                    x_sg, y_sg, z_sg = struct.unpack_from("<3f", val, 0)
                    sg_xyz = np.array([x_sg, y_sg, z_sg])
                    points.append(sg_xyz * geom.voxel_size + geom.ext_min)

            del db
            if points:
                return np.array(points, dtype=np.float32)
            return None

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    except Exception as exc:
        _log(f".sym counting points read failed: {exc}")
        return None


# -----------------------------------------------------------------------
# Per-label field construction
# -----------------------------------------------------------------------

def _prep_label(label_vol, label_id, sigma=0.0, margin=_PREP_MARGIN_VOXELS):
    """
    Build the signed uint8 field AddSurface meshes for ONE label, CROPPED to its bounding box
    (CPU-only, thread-safe — no COM/Ice access).

    Returns (field, (ox,oy,oz), n_painted, n_kept) or None if the label is empty.  Imaris reads
    the uint8 as signed int8 and meshes the zero crossing:
      sigma <= 0 : binary 100 (inside) / 200 (outside) — voxel fidelity (blocky surface,
                   nothing eroded).
      sigma  > 0 : a Gaussian-blurred gradient (+inside … −outside, 0 at the boundary), so
                   Imaris' marching cubes interpolates a smooth triangulated surface; larger
                   sigma = smoother (and rounds off / drops fine detail).
    `margin` grows with sigma so the blur has room to fall back to "outside" inside the crop.
    """
    mask = label_vol == label_id
    xs = np.any(mask, axis=(1, 2))
    if not xs.any():
        return None
    ys = np.any(mask, axis=(0, 2))
    zs = np.any(mask, axis=(0, 1))

    m = max(margin, int(np.ceil(3 * sigma)) + 1)   # room for the Gaussian tail

    def _span(flags, n):
        idx = np.nonzero(flags)[0]
        return max(0, int(idx[0]) - m), min(n, int(idx[-1]) + 1 + m)

    ox, x1 = _span(xs, mask.shape[0])
    oy, y1 = _span(ys, mask.shape[1])
    oz, z1 = _span(zs, mask.shape[2])

    sub = mask[ox:x1, oy:y1, oz:z1].copy()   # small crop; copy so the big bool can free
    del mask
    n_painted = int(sub.sum())

    if sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            blur = gaussian_filter(sub.astype(np.float32), sigma=sigma)
        except Exception:
            blur = _smooth_mask_3d(sub, iterations=max(1, int(round(2 * sigma))))
        signed = np.clip((blur - 0.5) * 200.0, -120, 120)     # +inside, −outside, 0 at surface
        field = signed.astype(np.int8).view(np.uint8)
    else:
        field = np.where(sub, 100, 200).astype(np.uint8)      # blocky, voxel fidelity

    n_kept = int((field.view(np.int8) > 0).sum())
    return field, (ox, oy, oz), n_painted, n_kept


def _smooth_mask_3d(mask, iterations=3):
    """
    Smooth a binary mask by iterative 7-point (self + 6 face-neighbour) averaging.

    Fallback for when scipy is unavailable; `iterations` stands in for a Gaussian sigma of
    roughly iterations/2.  Each pass is normalised by the actual neighbour count, so voxels
    on the array edge are not darkened toward zero the way a fixed divide by 7 would.

    Returns a float32 occupancy field in [0, 1] with the boundary at 0.5 — the same
    convention as scipy's gaussian_filter output, so the caller can rescale either one
    identically.
    """
    def _accumulate(a):
        n = a.copy()
        n[1:]       += a[:-1];      n[:-1]      += a[1:]
        n[:, 1:]    += a[:, :-1];   n[:, :-1]   += a[:, 1:]
        n[:, :, 1:] += a[:, :, :-1]; n[:, :, :-1] += a[:, :, 1:]
        return n

    m = mask.astype(np.float32)
    norm = _accumulate(np.ones_like(m))
    for _ in range(iterations):
        m = _accumulate(m) / norm
    return m


# -----------------------------------------------------------------------
# Upload to an IDataSet
# -----------------------------------------------------------------------

def _upload_bands(nx: int, ny: int, nz: int, budget: int | None = None):
    """
    Split an (nx, ny, nz) uint8 field into upload bands — the single source of truth for
    both the COM call pattern and the progress-bar tick count.

    Returns a list of (z0, z1, y_strips); each (z-range, y-strip) pair is one COM call and
    stays under `budget` bytes, and each band is one progress tick.  Whole z-bands are used
    when a full XY slice fits in the budget; otherwise one z-slice per band, tiled in Y.

    `budget` defaults to _ICE_CHUNK_BYTES read at call time, not import time, so the
    constant stays adjustable.
    """
    if budget is None:
        budget = _ICE_CHUNK_BYTES
    slice_bytes = nx * ny
    if slice_bytes <= budget:
        zb = max(1, budget // slice_bytes)
        return [(z0, min(nz, z0 + zb), [(0, ny)]) for z0 in range(0, nz, zb)]
    yb = max(1, budget // max(1, nx))
    strips = [(y0, min(ny, y0 + yb)) for y0 in range(0, ny, yb)]
    return [(z, z + 1, strips) for z in range(nz)]


def _upload_ticks(nx: int, ny: int, nz: int, budget: int | None = None) -> int:
    """Number of progress ticks _upload_field will emit for an (nx,ny,nz) uint8 field."""
    return len(_upload_bands(int(nx), int(ny), int(nz), budget))


def _upload_field(ds, field: np.ndarray, prog, budget: int | None = None) -> int:
    """
    Upload a cropped (nx,ny,nz) uint8 field to an IDataSet via SetDataSubVolumeAs1DArrayBytes.

    Imaris' 1D layout is X-fastest, then Y, then Z.  field is (X,Y,Z), so transposing a
    sub-block to (Z,Y,X) and calling the default C-order .tobytes() yields x-fastest bytes.
    Ticks `prog` once per band.  Returns the number of COM calls made.
    """
    nx, ny, nz = field.shape
    calls = 0
    for z0, z1, strips in _upload_bands(nx, ny, nz, budget):
        for y0, y1 in strips:
            blob = field[:, y0:y1, z0:z1].transpose(2, 1, 0).tobytes()
            ds.SetDataSubVolumeAs1DArrayBytes(blob, 0, y0, z0, 0, 0, nx, y1 - y0, z1 - z0)
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

    _log(f"no .syk next to the .ims (looked for {cand}); opening file picker.")
    return _ask_for_syk(ims_dir)


def _ask_for_syk(initialdir: str) -> str | None:
    """
    Open a modal file dialog for the .syk, starting in the open .ims's own directory.

    It deliberately does NOT remember the last-used directory: this runs on shared core
    workstations, where the previous user's folder is a worse starting guess than the
    folder the current .ims came from.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select the syGlass .syk mask file",
            initialdir=initialdir if os.path.isdir(initialdir) else None,
            filetypes=[("syGlass mask", "*.syk"), ("All files", "*.*")],
        )
        root.destroy()
        if path:
            return path
        _log("file picker cancelled.")
    except Exception as exc:
        _log(f"file dialog unavailable: {exc}")
    return None


# -----------------------------------------------------------------------
# Options menu
# -----------------------------------------------------------------------

# Surface-smoothing presets: label → Gaussian blur sigma (voxels).  0 = raw voxel fidelity.
_SMOOTHING_LEVELS = [
    ("None — voxel fidelity (blocky)", 0.0),
    ("Light", 0.8),
    ("Medium", 1.5),
    ("Strong", 2.5),
]


def _ask_settings() -> dict | None:
    """
    Show an options menu (tkinter) and return {'sigma': float, 'diagnostics': bool,
    'spots': bool}, or None if the user cancelled.

    Lets the user pick a surface-smoothing preset or type a custom blur, toggle
    troubleshooting (block diagnostics + mask preview in the log), and opt in to the
    experimental counting-point import.  Choices persist in the config so they default to
    last time.  Falls back to the saved defaults if no GUI is available.
    """
    cfg = _load_config()
    defaults = {"sigma": float(cfg.get("smoothing_sigma", 0.0)),
                "diagnostics": bool(cfg.get("diagnostics", False)),
                "spots": bool(cfg.get("spots", False))}
    try:
        import tkinter as tk
        root = tk.Tk()
        root.title("syGlass → Imaris — options")
        root.attributes("-topmost", True)

        tk.Label(root, text="Surface smoothing (Gaussian blur — smoother = less voxel detail):",
                 anchor="w").pack(fill="x", padx=14, pady=(12, 2))
        preset_vals = [v for _, v in _SMOOTHING_LEVELS]
        is_preset = defaults["sigma"] in preset_vals
        # A saved custom sigma leaves every radio unselected (-1 matches no preset) and
        # prefills the custom box, so the dialog shows what will actually be used.
        sigma_var = tk.DoubleVar(value=defaults["sigma"] if is_preset else -1.0)

        crow = tk.Frame(root)
        custom = tk.Entry(crow, width=8)

        def _clear_custom():
            """A preset click wins over a stale custom value (see _ok)."""
            custom.delete(0, "end")

        for name, val in _SMOOTHING_LEVELS:
            tk.Radiobutton(root, text=name, variable=sigma_var, value=val, anchor="w",
                           command=_clear_custom).pack(fill="x", padx=26)

        crow.pack(fill="x", padx=26, pady=(4, 2))
        tk.Label(crow, text="…or custom blur:").pack(side="left")
        custom.pack(side="left", padx=6)
        if not is_preset:
            custom.insert(0, str(defaults["sigma"]))

        diag_var = tk.BooleanVar(value=defaults["diagnostics"])
        tk.Checkbutton(root, text="Troubleshooting: log block diagnostics + mask preview",
                       variable=diag_var, anchor="w").pack(fill="x", padx=14, pady=(12, 2))

        spots_var = tk.BooleanVar(value=defaults["spots"])
        tk.Checkbutton(root, text="EXPERIMENTAL: import .sym counting points (positions unverified)",
                       variable=spots_var, anchor="w").pack(fill="x", padx=14, pady=(0, 2))

        out = {}
        state = {"cancelled": False}

        def _ok():
            s = float(sigma_var.get())
            c = custom.get().strip()
            if c:
                try:
                    s = float(c)
                except ValueError:
                    pass
            out["sigma"] = max(0.0, s)
            out["diagnostics"] = bool(diag_var.get())
            out["spots"] = bool(spots_var.get())
            root.quit()

        def _cancel():
            state["cancelled"] = True
            root.quit()

        brow = tk.Frame(root); brow.pack(pady=12)
        tk.Button(brow, text="Run", width=12, command=_ok).pack(side="left", padx=6)
        tk.Button(brow, text="Cancel", width=12, command=_cancel).pack(side="left", padx=6)
        root.protocol("WM_DELETE_WINDOW", _cancel)
        root.mainloop()
        root.destroy()

        if state["cancelled"]:
            return None

        result = out or defaults
        cfg["smoothing_sigma"] = result["sigma"]
        cfg["diagnostics"] = result["diagnostics"]
        cfg["spots"] = result["spots"]
        _save_config(cfg)
        return result
    except Exception as exc:
        _log(f"options menu unavailable ({exc}); using saved defaults {defaults}")
        return defaults


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
            _log(f"progress GUI unavailable ({exc}); using console output.")

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
        self.every = 1
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

def _log_warning(message: str) -> None:
    """Show a warning dialog and log the same message."""
    _warn(message)
    try:
        import tkinter.messagebox
        tkinter.messagebox.showwarning("Import from Syglass:", message)
    except Exception:
        pass
