"""
Imaris XTension: Import syGlass Masks into Imaris.

Runs inside a live Imaris session. Imaris calls XTImportFromSyGlass(imarisFile)
where imarisFile is the numeric handle for the ImarisLib COM connection.

Usage
-----
1. Open the .ims file in Imaris.
2. Run Extensions → Import from syGlass.  The matching .syk is found automatically
   if it sits beside the .ims; otherwise a file picker opens so it can be anywhere.

Strategy
--------
Masks:
  1. Parse the .syk octree directly, rather than going through the `syglass` Python API
     (which turned out to be more trouble than it was worth for these files). Block
     headers are scanned to inventory the octree; the root block (a whole-volume preview
     downsampled by 2**max_level) gives a cheap bounding-box estimate; every region inside
     that box with no stored finer block — whole childless blocks, or the octants of a
     partly subdivided one, either of which may sit above the deepest level — is then
     read and composited; empty placeholder blocks don't count as stored.  Blocks are stored (CZ, CY, CX) and overlap
     their neighbours; see _DEFAULT_TRIM for the per-axis overlap.
  2. Split each label into its disconnected components, and build a signed uint8 field
     per component, cropped to the component's bounding box (sigma <= 0: binary
     100=inside / 200=outside; sigma > 0: a clipped Gaussian gradient). Each field is
     uploaded in bands via SetDataSubVolumeAs1DArrayBytes, each call kept under the Ice
     message limit.
  3. Call ISurfaces.AddSurface once per component, all into one ISurfaces item per label:
     the scene tree shows one entry per label, but inside it every component is its own
     surface object, so debris is selectable and size-filterable in Imaris (Filter tab,
     e.g. "Number of Voxels"). Prep for the next label runs on a worker thread while the
     current one uploads.

Progress + logs:
  A tkinter window shows the estimated finish time.  A timestamped log is written to a
  'logs' folder inside the script's own directory (…/<script_dir>/logs/import_from_syglass_xt_*.log).

Limitations:
  Single timepoint. Multichannel is fine.  Counting points are not imported — a prior
  experimental .sym reader is in git history if that is ever picked up again.

Troubleshooting:
    - Ensure that the directory containing this script is in the Imaris Preferences CustomTools → Python path list
    - Check the log file for errors.
    - Turn on diagnostics in the options menu to see ASCII max-projections of the reconstructed mask (shape sanity check, independent of Imaris).
    - If the import appears hung, check Task Manager: both Python and Imaris should show CPU activity while surfaces are built.

Python 3.11 compatibility required (Imaris bundled interpreter).

Copyright (C) 2026 Mark Becker

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version. See the LICENSE file for details.
"""

from __future__ import annotations

import datetime
import os
import queue
import struct
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field as dataclass_field

import numpy as np

# SciPy is optional and imported once here.  If the import fails, the NumPy fallbacks
# are used and _run writes the reason to the log.
try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian_filter
    from scipy.ndimage import label as _scipy_label
    _SCIPY_IMPORT_ERROR = None
except ImportError as _scipy_exc:
    _scipy_gaussian_filter = None
    _scipy_label = None
    _SCIPY_IMPORT_ERROR = _scipy_exc

# ImarisLib — always available in Imaris's Python interpreter
import ImarisLib  # type: ignore  # noqa: E402


# -----------------------------------------------------------------------
# Tunables
# -----------------------------------------------------------------------

# Magic at the start of every .syk block record.
_FVGU = b"fvgu"

# Largest payload (bytes) to push in a single COM call.  Ice's default MessageSizeMax
# is ~1 MB; staying under it lets us upload a band of z-slices per call.
_ICE_CHUNK_BYTES = 900_000

# syGlass label IDs are generally small.  Anything above this is probably a storage
# artifact rather than a painted label, and is erased at read time so it cannot show up as
# a phantom surface.
_MAX_PLAUSIBLE_LABEL = 100

# Voxels dropped from the trailing edge of each block per axis, i.e. block_dim - stride.
#
# Found by rendering real files and comparing the result, which turned out to be the only
# reliable test: (3,1,1) and (2,1,1) both give clean surfaces, while (0,0,0) reproduces the
# periodic displaced-slice artifact.  So blocks do overlap their neighbours, by 1 voxel in
# Y and Z and by 2-3 in X.
#
# The syGlass grid doesn't have to line up with the .ims voxel grid — the mask gets scaled
# onto the .ims extents — so a grid slightly smaller than the image is expected, not a sign
# these numbers are wrong.  A couple of earlier attempts to infer them from file structure
# went astray by assuming otherwise.  The options menu exposes them for testing against a
# new file.
_DEFAULT_TRIM = (3, 1, 1)

# Extra root-level blocks of margin on each side of the estimated mask bounding box.
_ROOT_BBOX_MARGIN_BLOCKS = 2

# Minimum crop padding (voxels) around a label's bounding box, so a blurred field has
# room to fall back to "outside" without touching the crop wall.
_PREP_MARGIN_VOXELS = 4

# Above this many components in one label, log a warning that per-component AddSurface
# calls will be slow.  Log-only, not a dialog.
_COMPONENT_WARN_THRESHOLD = 200

# Padding on the finish-time estimate, as a fraction of the projected remaining time.
# It is applied in full at the start of the build and shrinks to zero as the build
# completes, so early estimates lean late and settle as more surfaces finish.
_ETA_PESSIMISM = 0.5

# Open log file handle (set by _setup_logging); every _log() line carries a timestamp
# so it is clear which step consumes time.  None until setup, in which case _log()
# still prints to the console.
_LOG_FH = None

# Messages passed to _warn during the current run, shown again on the summary page.
_RUN_WARNINGS: list[str] = []


class _SykLockedError(RuntimeError):
    """The .syk may be held open by syGlass (Windows sharing violation → PermissionError)."""


# How to release a locked .syk — shared by the retry dialog and the give-up warning.
# Two steps because syGlass has been seen keeping the file handle open even after the
# project itself is closed.
_LOCKED_SYK_HINT = ("Close the project in syGlass. If the file is still locked after "
                    "that, exit syGlass entirely.")


# -----------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------

def _setup_logging():
    """
    Open a timestamped log file in a 'logs' folder inside the script's own directory —
    i.e. <script_dir>/logs/import_from_syglass_xt_<timestamp>.log.  Returns the path, or None on failure.
    """
    global _LOG_FH
    _RUN_WARNINGS.clear()
    sys.stderr = _StderrToLog(sys.__stderr__)
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
    """Log a line the user should notice, and keep it for the summary page."""
    _RUN_WARNINGS.append(msg)
    _log(f"WARNING: {msg}")


class _StderrToLog:
    """
    Stand-in for sys.stderr that writes to the console and copies the text to the log.
    Libraries print warnings and tracebacks to stderr, and with the console minimized
    they would otherwise be easy to miss.
    """

    def __init__(self, console) -> None:
        self.console = console

    def write(self, text: str) -> int:
        self.console.write(text)
        if _LOG_FH is not None:
            _LOG_FH.write(text)
        return len(text)

    def flush(self) -> None:
        self.console.flush()

    def __getattr__(self, name):
        # Anything else (encoding, isatty, fileno) comes from the real console stream.
        return getattr(self.console, name)


def _minimize_console() -> None:
    """Minimize the console window of the XTension's Python process (Windows only)."""
    if sys.platform != "win32":
        return
    import ctypes
    console_window = ctypes.windll.kernel32.GetConsoleWindow()
    if console_window:
        show_minimized = 6    # SW_MINIMIZE
        ctypes.windll.user32.ShowWindow(console_window, show_minimized)


# -----------------------------------------------------------------------
# XTension entry point (Imaris calls this)
# -----------------------------------------------------------------------

#   <CustomTools>
#       <Menu>
#           <Item name="Import from syGlass" icon="Python3"
#                 tooltip="Import syGlass masks into the current scene as surfaces">
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
    _minimize_console()

    vLib = ImarisLib.ImarisLib()
    vApp = vLib.GetApplication(imarisFile)

    # ----------------------------------------------------------------
    # 1. Locate companion syGlass files
    # ----------------------------------------------------------------
    ims_path = vApp.GetCurrentFileName()
    if not ims_path:
        _warn_dialog("No file is currently open in Imaris.")
        return

    ims_dir  = os.path.dirname(ims_path)
    ims_stem = os.path.splitext(os.path.basename(ims_path))[0]

    # Locate the .syk: try the conventional same-dir/same-stem path first, then a
    # case-insensitive scan, then fall back to a GUI file picker (the .syk may live
    # elsewhere or be named differently from the .ims).
    syk_path = _resolve_syk_path(ims_dir, ims_stem)
    if not syk_path or not os.path.exists(syk_path):
        _warn_dialog("No .syk file selected — aborting import.")
        return

    _log(f"using .syk: {syk_path}")

    # Fail fast on a locked .syk — before the options dialog, not after minutes of work.
    if not _ensure_syk_readable(syk_path, _ask_retry_dialog):
        _log("import aborted — .syk locked by syGlass.")
        return

    # From here on, everything the user sees happens in one window: options, progress,
    # then a summary that stays open until closed.  A failure also ends on the summary.
    window = _Window()
    try:
        _import(vApp, syk_path, ims_stem, window, log_path)
    except Exception as exc:
        _log(traceback.format_exc())
        window.show_summary(_Summary(
            succeeded=False,
            headline="The import failed.",
            details=[f"{type(exc).__name__}: {exc}"],
            warnings=list(_RUN_WARNINGS),
            log_path=log_path))
        raise


def _import(vApp, syk_path: str, ims_stem: str, window: _Window, log_path: str | None) -> None:
    """Options, mask read and surface build, reported through `window`."""
    settings = window.ask_options()
    if settings is None:
        _log("options menu cancelled — nothing imported.")
        window.close()
        return
    _log(f"options: smoothing sigma={settings.sigma}  "
         f"diagnostics={settings.diagnostics}  "
         f"block trim={settings.trim}")
    if _SCIPY_IMPORT_ERROR is not None:
        _warn(f"scipy could not be imported ({_SCIPY_IMPORT_ERROR}); using the NumPy "
              f"fallbacks. Component labeling is slower, and smoothing approximates the "
              f"Gaussian blur, so surfaces differ from a run with scipy.")

    # ----------------------------------------------------------------
    # 2. Read image dimensions and physical extents from Imaris
    # ----------------------------------------------------------------
    geom = _Geometry(vApp.GetDataSet())
    for line in geom.describe():
        _log(line)
    if geom.size_t > 1:
        _warn(f".ims has {geom.size_t} timepoints; surfaces are built from a single volume "
              f"and will land on t=0 only.")

    def stop(message: str) -> None:
        """End the import early, showing why on the summary page."""
        _log(f"import stopped: {message}")
        window.show_summary(_Summary(
            succeeded=False,
            headline="Nothing was imported.",
            details=[message],
            warnings=list(_RUN_WARNINGS),
            log_path=log_path))

    # ----------------------------------------------------------------
    # 3. Read syGlass label mask
    # ----------------------------------------------------------------
    status = _ImportStatus()
    window.show_progress(status)

    status.begin_stage("read")
    t_read0 = time.time()

    def block_read(done: int, total: int) -> None:
        status.set_activity(f"Reading mask blocks ({done} of {total})", done, total)

    # label_vol: (NX, NY, NZ) uint16, 0=background, 1..N=label.
    # clip_info: (vx0, vy0, vz0, full_nx, full_ny, full_nz) — the sub-region that was read.
    # syGlass can re-acquire the lock between the pre-flight check and here, so the read
    # itself sits in the same retry loop.
    while True:
        try:
            label_vol, clip_info = _read_mask_from_syk(syk_path,
                                                       diagnostics=settings.diagnostics,
                                                       trim=settings.trim,
                                                       on_block=block_read)
            break
        except _SykLockedError:
            if not _ensure_syk_readable(syk_path, window.ask_retry):
                stop(f"The .syk is locked by syGlass. {_LOCKED_SYK_HINT} "
                     f"Then run the import again.")
                return

    if label_vol is None or not np.any(label_vol):
        stop("No mask data was found in the .syk. The file is empty or could not be "
             "read; the log has details.")
        return
    _log(f"mask read in {time.time() - t_read0:.1f}s")

    status.begin_stage("labels")
    status.set_activity("Counting voxels per label")
    t_scan0 = time.time()
    label_ids, counts = _inventory_labels(label_vol)
    _log(f"label_vol shape: {label_vol.shape}, dtype: {label_vol.dtype}")
    _log(f"Non-zero voxels: {int(label_vol.size - counts[0])} / {label_vol.size}  "
         f"(inventory in {time.time() - t_scan0:.1f}s)")
    for v in label_ids:
        _log(f"  label {v}: {int(counts[v])} voxels")
    _log(f"Found {len(label_ids)} labels: {label_ids}")
    if not label_ids:
        stop("The .syk contains voxel data but no usable label IDs, so there is nothing "
             "to mesh.")
        return

    # Troubleshooting only: ASCII max-projections of the reconstructed mask (shape sanity
    # check, independent of Imaris).
    if settings.diagnostics:
        status.set_activity("Rendering mask preview for the log")
        _log_mask_projections(label_vol)

    # ----------------------------------------------------------------
    # 4. Build one ISurfaces per label
    # ----------------------------------------------------------------
    factory = vApp.GetFactory()
    scene   = vApp.GetSurpassScene()
    status.begin_stage("build")
    result = _build_surfaces(factory, scene, label_vol, label_ids,
                             [int(counts[label_id]) for label_id in label_ids],
                             geom, clip_info, settings.sigma, status, ims_stem)
    status.begin_stage("done")

    # ----------------------------------------------------------------
    # 5. Done — not saving on purpose.
    #    vApp.FileSave("", "") pops Imaris's Save dialog (empty filename) and rewrites the
    #    whole .ims (~3 min on large files).  The surfaces are already in the live Surpass
    #    scene, so saving is left to the user (Ctrl+S) whenever they're ready.
    # ----------------------------------------------------------------
    _log("Done — surfaces added to the scene. Press Ctrl+S "
         "in Imaris to write them to the .ims when ready.")

    details = [f"Created {result.n_items} surface item(s) holding {result.n_objects} "
               f"object(s) in {_format_duration(time.time() - t_read0)}.",
               "The surfaces are in the scene but not saved. Press Ctrl+S in Imaris to "
               "write them to the .ims."]
    if result.n_vanished:
        details.append(f"{result.n_vanished} small piece(s) ({result.vanished_voxels} "
                       f"voxels) disappeared under the smoothing and were skipped. "
                       f"Lower the smoothing to keep them.")
    window.show_summary(_Summary(
        succeeded=True,
        headline="Import finished.",
        details=details,
        warnings=list(_RUN_WARNINGS),
        log_path=log_path))


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


def _field_extents(ds_lo, per_voxel, offset, crop):
    """
    Physical extents (µm) to set on a crop's IDataSet so AddSurface lands the field
    exactly on the voxel grid.

    AddSurface places field samples right on the extent borders: sample i sits at
    extMin + i·(extMax − extMin)/(size − 1)  (ISurfaces::AddSurface: "voxel size is
    (mExtentMax - mExtentMin) / (mSize - 1)").  But field sample i actually holds the
    value at the center of voxel i, i.e. at lo_edge + (i + 0.5)·per_voxel — so each
    extent needs to be inset by half a voxel to make the two line up, which gives a
    sample spacing of exactly per_voxel.  Using the raw outer edges instead stretches
    the surface by size/(size−1) and shifts it half a voxel — worst on small crops, and
    a different amount for every crop size.
    """
    offset = np.asarray(offset, dtype=np.float64)
    crop = np.asarray(crop, dtype=np.float64)
    lo_edge = ds_lo + offset * per_voxel
    hi_edge = ds_lo + (offset + crop) * per_voxel
    return lo_edge + 0.5 * per_voxel, hi_edge - 0.5 * per_voxel


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
    slab = 64
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


@dataclass
class _BuildResult:
    """What _build_surfaces created, for the summary page."""
    n_items: int
    n_objects: int
    n_vanished: int
    vanished_voxels: int


def _build_surfaces(factory, scene, label_vol, label_ids, label_voxels: list[int],
                    geom: _Geometry, clip_info, sigma: float, status: _ImportStatus,
                    name_prefix: str) -> _BuildResult:
    """
    Create one ISurfaces item per label — holding one surface object per disconnected
    component — and add it to the Surpass scene.  `label_voxels` holds each label's voxel
    count, in the order of `label_ids`, for the finish-time estimate in `status`.

    AddSurface interprets the dataset as a signed field and finds the surface at the zero
    crossing (uint8 treated as signed int8: 100 → +100 inside, 200 → −56 outside), and
    each call yields exactly one surface object no matter how many blobs the field
    contains.  So every component gets its own IDataSet, sized to that component's own
    bounding box (not the full reconstruction, which may be billions of mostly-empty
    voxels), and its own AddSurface call; that's what makes debris individually
    selectable and size-filterable in Imaris.  Data is uploaded in chunks kept under the
    Ice limit.

    If per-component AddSurface round-trips ever prove too slow on debris-heavy labels,
    an alternative is to relabel components as distinct values and make one
    IImageProcessing.DetectSurfacesFromLabelImage call instead — but that bypasses the
    signed field, so the surfaces come out blocky.  Not worth it unless timing says
    otherwise.

    Prefetch depth 1: prep for the next label (crop + split + smooth + field build) runs
    on a background thread while the main thread uploads the current one, so the
    pure-NumPy prep overlaps the serial COM I/O.
    """
    eTypeUInt8 = _resolve_uint8_type()
    vol_shape = np.array(label_vol.shape)
    ds_lo, ds_hi, full_grid = _clip_extents(geom, clip_info, vol_shape)

    if clip_info is not None:
        _log(f"IDataSet (clipped): {vol_shape[0]}×{vol_shape[1]}×{vol_shape[2]} voxels  "
             f"full grid {int(full_grid[0])}×{int(full_grid[1])}×{int(full_grid[2])}")
    _log(f"IDataSet extents: X=[{ds_lo[0]:.1f}, {ds_hi[0]:.1f}] "
         f"Y=[{ds_lo[1]:.1f}, {ds_hi[1]:.1f}] Z=[{ds_lo[2]:.1f}, {ds_hi[2]:.1f}]")

    # The syGlass grid need not match the .ims voxel grid: the mask is mapped onto the
    # .ims extents proportionally, so syGlass is free to build its octree at whatever
    # resolution suits it.  A ratio near but not equal to 1.0 is normal; it's logged as
    # context, not as a sign of a problem.
    ratios = geom.size / full_grid
    _log(f"ims/syk grid ratio: X={ratios[0]:.3f}  Y={ratios[1]:.3f}  Z={ratios[2]:.3f}")

    # Physical size of one clip voxel — converts each label's crop (in clip-voxel coords)
    # into the µm extents of its own IDataSet.
    per_voxel = (ds_hi - ds_lo) / vol_shape

    n_labels = len(label_ids)
    _log(f"{n_labels} label(s); each cropped to its bbox, "
         f"uploaded in <= {_ICE_CHUNK_BYTES // 1000} KB chunks; prep pipelined 1 ahead")

    t_run0 = time.time()
    sum_prep = sum_upload = sum_detect = 0.0
    n_items = n_objects = 0
    n_vanished = vanished_voxels = 0

    status.n_labels = n_labels
    status.label_voxels = list(label_voxels)

    def label_prefix(label_idx: int) -> str:
        if n_labels == 1:
            return ""
        return f"Label {label_idx + 1} of {n_labels}: "

    def prep_reporter(label_idx: int):
        """
        Progress callback for one label's prep.  Prep runs one label ahead, so its
        updates only reach the window while the build loop is waiting for that label.
        """
        def report(text: str, done: int, total: int) -> None:
            if status.waiting_for_prep and status.label_index == label_idx:
                status.set_activity(label_prefix(label_idx) + text, done, total)
        return report

    # One worker is enough: the loop only submits the next prep after collecting the
    # current one, so at most one prep is ever in flight.
    with ThreadPoolExecutor(max_workers=1) as pool:
        next_future = pool.submit(_prep_label, label_vol, label_ids[0], sigma,
                                  on_progress=prep_reporter(0))
        for label_idx, label_id in enumerate(label_ids):
            tag = f"L{label_idx + 1}/{n_labels} (id {label_id})"

            status.begin_label(label_idx)
            status.waiting_for_prep = True
            status.set_activity(label_prefix(label_idx) + "Preparing surfaces")
            t0 = time.time()
            prep = next_future.result()
            t_prep = time.time() - t0
            status.waiting_for_prep = False
            # Kick off prep for the next label so it overlaps this label's upload.
            if label_idx + 1 < n_labels:
                next_future = pool.submit(_prep_label, label_vol,
                                          label_ids[label_idx + 1], sigma,
                                          on_progress=prep_reporter(label_idx + 1))

            if prep is None or not prep["components"]:
                if prep is None:
                    _log(f"{tag}: empty — skipped")
                else:
                    _log(f"{tag}: all {prep['n_components']} component(s) vanished "
                         f"under the blur — skipped")
                    n_vanished += prep["n_vanished"]
                    vanished_voxels += prep["vanished_voxels"]
                continue

            comps = prep["components"]
            n_comps = len(comps)
            _log(f"{tag}: {n_comps}/{prep['n_components']} component(s); "
                 f"painted={prep['n_painted']}")
            if prep["n_vanished"]:
                _log(f"{tag}: {prep['n_vanished']} component(s) vanished under the blur "
                     f"({prep['vanished_voxels']} vox) — too small for sigma={sigma}; "
                     f"lower the smoothing to keep them")
                n_vanished += prep["n_vanished"]
                vanished_voxels += prep["vanished_voxels"]
            if n_comps > _COMPONENT_WARN_THRESHOLD:
                _warn(f"{tag}: {n_comps} disconnected components — one AddSurface call "
                      f"each will be slow; debris can be deleted afterwards in Imaris "
                      f"(Filter tab, e.g. \"Number of Voxels\")")

            status.pieces_in_label = n_comps
            surfaces = factory.CreateSurfaces()

            t_up = t_det = 0.0
            for ci, (field, offset, comp_voxels, n_kept) in enumerate(comps):
                status.set_activity(label_prefix(label_idx) +
                                    f"Building surface {ci + 1} of {n_comps} in Imaris",
                                    ci, n_comps)
                crop = np.array(field.shape)
                # Physical extents of this component's crop (sub-region of the clip volume).
                lo, hi = _field_extents(ds_lo, per_voxel, offset, crop)
                _log(f"{tag} c{ci + 1}/{n_comps}: crop {crop[0]}x{crop[1]}x{crop[2]} "
                     f"({crop.prod() / 1e6:.1f} MB) at voxel "
                     f"({offset[0]},{offset[1]},{offset[2]}); voxels={comp_voxels} "
                     f"kept={n_kept}; bbox um X=[{lo[0]:.0f},{hi[0]:.0f}] "
                     f"Y=[{lo[1]:.0f},{hi[1]:.0f}] Z=[{lo[2]:.0f},{hi[2]:.0f}]")

                sdf_ds = factory.CreateDataSet()
                sdf_ds.Create(eTypeUInt8, int(crop[0]), int(crop[1]), int(crop[2]), 1, 1)
                sdf_ds.SetExtendMinX(lo[0]); sdf_ds.SetExtendMinY(lo[1]); sdf_ds.SetExtendMinZ(lo[2])
                sdf_ds.SetExtendMaxX(hi[0]); sdf_ds.SetExtendMaxY(hi[1]); sdf_ds.SetExtendMaxZ(hi[2])

                t0 = time.time()
                _upload_field(sdf_ds, field)
                t_up += time.time() - t0

                # One AddSurface per component = one surface object per component.
                t0 = time.time()
                surfaces.AddSurface(sdf_ds, 0)
                t_det += time.time() - t0
                status.piece_built(comp_voxels)
            del comps, prep   # free this label's fields before the next prep is awaited

            try:
                n_surf = surfaces.GetNumberOfSurfaces()
            except Exception as exc:
                n_surf = f"?({exc})"

            r, g, b = _DEFAULT_PALETTE[(label_id - 1) % len(_DEFAULT_PALETTE)]
            surfaces.SetColorRGBA(_pack_rgba(r, g, b, 255))
            surfaces.SetName(f"{name_prefix} label {label_id}")
            scene.AddChild(surfaces, -1)
            n_items += 1
            n_objects += n_comps

            sum_prep += t_prep; sum_upload += t_up; sum_detect += t_det
            _log(f"{tag}: prep_wait={t_prep:.2f}s  upload={t_up:.2f}s  "
                 f"detect={t_det:.2f}s  surfaces_in_object={n_surf}")

    t_total = time.time() - t_run0
    _log(f"Created {n_items} surface item(s) holding {n_objects} object(s) "
         f"in {t_total:.1f}s  "
         f"(prep_wait {sum_prep:.1f}s + upload {sum_upload:.1f}s + detect {sum_detect:.1f}s; "
         f"avg {t_total / max(1, n_labels):.2f}s/label)")
    return _BuildResult(n_items=n_items, n_objects=n_objects,
                        n_vanished=n_vanished, vanished_voxels=vanished_voxels)


# -----------------------------------------------------------------------
# Mask reader: manual .syk parser (main)
# -----------------------------------------------------------------------

def _read_mask_from_syk(syk_path: str, diagnostics: bool = False,
                        trim=_DEFAULT_TRIM, on_block=None):
    """
    Read the syGlass label mask from a .syk file.

    Scans all block headers to inventory the octree, reads the root block to
    estimate the mask bounding box, then composites, within that bbox, every
    region that has no stored finer block: childless blocks whole, and the
    child-less octants of partly subdivided ones (coarse data is upsampled).
    Reading only the bbox avoids allocating the full (potentially huge)
    reconstructed volume.

    `on_block(done, total)`, if given, is called as blocks are visited.  Each block in
    the bbox is visited twice, once to check it for mask and once to place it.

    Returns ((NX, NY, NZ) uint16 array, clip_info) where
      clip_info = (vx0, vy0, vz0, full_nx, full_ny, full_nz)
    or (None, None) on failure.  Raises _SykLockedError if the file is held open by
    syGlass, so the caller can offer a retry instead of reporting an empty file.
    """
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

            block_offset = _scan_block_headers(f, n_blocks_est, block_stride, block_payload)
            if not block_offset:
                _log(".syk: no valid blocks found")
                return None, None

            max_level = max(_syk_block_level(bid) for bid in block_offset)
            n_grid    = 2 ** max_level
            # ax/ay/az = voxels kept per block per axis, which is also the placement
            # stride: blocks overlap their neighbours by `trim` voxels and the trailing
            # ones are discarded.  See _DEFAULT_TRIM for how the values were established
            # and what is still unknown about them.
            tx, ty, tz = (int(v) for v in trim)
            ax, ay, az = cx - tx, cy - ty, cz - tz
            if min(ax, ay, az) < 1:
                _log(f".syk: block trim {trim} is larger than the block; aborting")
                return None, None
            full_nx, full_ny, full_nz = n_grid * ax, n_grid * ay, n_grid * az

            _log(f".syk: {len(block_offset)} blocks; "
                 f"deepest level={max_level}  full grid={full_nx}×{full_ny}×{full_nz}")

            bbox = _estimate_bbox(f, block_offset, (cx, cy, cz), block_payload,
                                  n_grid, (full_nx, full_ny, full_nz))
            (bbox_x0, bbox_y0, bbox_z0), (bbox_x1, bbox_y1, bbox_z1) = bbox

        clip_nx = bbox_x1 - bbox_x0
        clip_ny = bbox_y1 - bbox_y0
        clip_nz = bbox_z1 - bbox_z0
        _log(f".syk: allocating clipped vol {clip_nx}×{clip_ny}×{clip_nz} "
             f"(origin {bbox_x0},{bbox_y0},{bbox_z0})")

        # uint16 (not uint32): the .syk payload is already uint16 and label IDs fit, so
        # this halves the footprint of the volume and the bandwidth of every scan over it.
        vol = np.zeros((clip_nx, clip_ny, clip_nz), dtype=np.uint16)

        # Pass 2 — place every region within the bbox that has no stored finer block.
        #
        # The octree is adaptive: a block whose children are absent from the file is a leaf
        # holding its region's mask at its own (coarser) resolution.  Reading only the
        # deepest level therefore drops data — on one test file 478 of 512 level-3 blocks
        # were childless and carried mask, so all but 34 regions came out empty.  Whether a
        # given file is adaptive or uniform varies: a second file stored everything at the
        # deepest level and read correctly either way.  tools/octree_structure.py reports
        # which shape a file has.
        #
        # A block can also be partly subdivided: children stored for some octants and not
        # others.  The octants without a stored child exist only in that coarse block, so
        # they are painted from it; skipping the whole block because it has any children
        # cut the mask off at block boundaries on a real file.  A childless block is the
        # case where all eight octants come from the block itself.
        #
        # syGlass also stores empty placeholder blocks: no mask voxel at all, under a parent
        # whose matching octant does hold mask.  Parents otherwise track their children
        # closely (a parent octant is close to a downsample of its child), so an empty
        # child is not an erasure — treating it as authoritative left a whole block row of
        # a real file blank.  Such placeholders are therefore transparent: they paint
        # nothing and do not count as stored children, so their region comes from the
        # nearest non-empty ancestor.  Blocks are painted coarse to fine, so a non-empty
        # block below a placeholder still overwrites the ancestor's data in its region.
        #
        # A block at level L is coarser than the finest grid by scale = 2**(max_level - L),
        # so each of its voxels covers a scale**3 cube and is expanded on placement.
        def read_block(f, boff):
            f.seek(boff + 24)
            raw = f.read(block_payload)
            # Stored (cz, cy, cx).  Keep the leading (ax, ay, az) voxels — the trailing
            # `trim` slices per axis are dropped as overlap/padding — then transpose to the
            # (X, Y, Z) convention.
            return (np.frombuffer(raw, dtype="<u2")
                    .reshape(cz, cy, cx)[:az, :ay, :ax]
                    .transpose(2, 1, 0))   # (ax, ay, az), uint16

        # Geometry of every block that overlaps the bbox, in finest-grid voxels.  A child
        # outside the bbox has a parent octant outside it too, so nothing else matters.
        geom = {}
        for bid in block_offset:
            lv, ix, iy, iz = _syk_block_position(bid)
            scale = 1 << (max_level - lv)     # finest voxels per voxel of this block
            span_x, span_y, span_z = ax * scale, ay * scale, az * scale
            bx0, by0, bz0 = ix * span_x, iy * span_y, iz * span_z
            if bx0 + span_x <= bbox_x0 or bx0 >= bbox_x1: continue
            if by0 + span_y <= bbox_y0 or by0 >= bbox_y1: continue
            if bz0 + span_z <= bbox_z0 or bz0 >= bbox_z1: continue
            geom[bid] = (lv, scale, (span_x, span_y, span_z), (bx0, by0, bz0))

        def has_mask(a):
            return bool(((a > 0) & (a <= _MAX_PLAUSIBLE_LABEL)).any())

        blocks_to_visit = 2 * len(geom)
        blocks_visited = 0

        with open(syk_path, "rb") as f:
            empty = set()
            for bid in geom:
                if not has_mask(read_block(f, block_offset[bid])):
                    empty.add(bid)
                blocks_visited += 1
                if on_block is not None:
                    on_block(blocks_visited, blocks_to_visit)

            # An empty block is a placeholder only if its nearest non-empty ancestor has
            # mask in the voxels that lie wholly inside the block's region.  The trimmed
            # block size is odd, so an ancestor voxel can straddle the region's edge and
            # carry mask from next door; counting those would turn a genuinely empty
            # block into a one-voxel sheet along its face.
            ancestor_cache = {}
            placeholder = set()
            for bid in empty:
                anc = bid
                while anc:
                    anc = (anc - 1) // 8
                    if anc in geom and anc not in empty:
                        break
                else:
                    continue                      # no non-empty ancestor: root is empty
                if anc not in ancestor_cache:
                    ancestor_cache[anc] = read_block(f, block_offset[anc])
                _, a_scale, _, a_org = geom[anc]
                _, _, spans, org = geom[bid]
                inner = tuple(slice(-(-(o - ao) // a_scale), (o + s - ao) // a_scale)
                              for o, s, ao in zip(org, spans, a_org))
                if has_mask(ancestor_cache[anc][inner]):
                    placeholder.add(bid)
            ancestor_cache.clear()

            children = {}
            for bid in geom:
                if bid and bid not in placeholder:
                    children.setdefault((bid - 1) // 8, set()).add((bid - 1) % 8)

            n_placed = 0
            n_partial = 0
            n_sentinel = 0
            per_level = {}
            for bid in sorted(geom, key=lambda b: geom[b][0]):
                blocks_visited += 1
                if on_block is not None:
                    on_block(blocks_visited, blocks_to_visit)
                if bid in placeholder:
                    continue                      # its ancestor holds the data
                stored = children.get(bid, set())
                if len(stored) == 8:
                    continue                      # interior node: its children hold the data
                lv, scale, (span_x, span_y, span_z), (bx0, by0, bz0) = geom[bid]
                arr = read_block(f, block_offset[bid])

                # Regions of this block to paint, in block-local finest-grid voxels: the
                # whole block for a leaf, else each octant with no stored child.  A block
                # with children sits above max_level, so scale >= 2 and the halves are exact.
                if not stored:
                    regions = [((0, span_x), (0, span_y), (0, span_z))]
                else:
                    hx, hy, hz = span_x // 2, span_y // 2, span_z // 2
                    regions = [((hx * (o & 1), hx * (o & 1) + hx),
                                (hy * (o >> 1 & 1), hy * (o >> 1 & 1) + hy),
                                (hz * (o >> 2 & 1), hz * (o >> 2 & 1) + hz))
                               for o in range(8) if o not in stored]
                    n_partial += 1

                for (rx0, rx1), (ry0, ry1), (rz0, rz1) in regions:
                    # Clip the region to the bbox, still block-local.  Named src_/dst_
                    # rather than c*/d* because cx, cy, cz are the block dimensions and in
                    # scope here.
                    src_x0 = max(rx0, bbox_x0 - bx0);  src_x1 = min(rx1, bbox_x1 - bx0)
                    src_y0 = max(ry0, bbox_y0 - by0);  src_y1 = min(ry1, bbox_y1 - by0)
                    src_z0 = max(rz0, bbox_z0 - bz0);  src_z1 = min(rz1, bbox_z1 - bz0)
                    if src_x0 >= src_x1 or src_y0 >= src_y1 or src_z0 >= src_z1:
                        continue

                    # Crop at the block's own resolution before upsampling, so a coarse
                    # block near the root never expands to its full finest-grid size.
                    piece = arr[src_x0 // scale:-(-src_x1 // scale),
                                src_y0 // scale:-(-src_y1 // scale),
                                src_z0 // scale:-(-src_z1 // scale)]
                    if scale > 1:
                        piece = piece.repeat(scale, 0).repeat(scale, 1).repeat(scale, 2)
                        ox, oy, oz = src_x0 % scale, src_y0 % scale, src_z0 % scale
                        piece = piece[ox:ox + src_x1 - src_x0,
                                      oy:oy + src_y1 - src_y0,
                                      oz:oz + src_z1 - src_z0]

                    dst_x0 = bx0 + src_x0 - bbox_x0;  dst_x1 = dst_x0 + (src_x1 - src_x0)
                    dst_y0 = by0 + src_y0 - bbox_y0;  dst_y1 = dst_y0 + (src_y1 - src_y0)
                    dst_z0 = bz0 + src_z0 - bbox_z0;  dst_z1 = dst_z0 + (src_z1 - src_z0)

                    dst = vol[dst_x0:dst_x1, dst_y0:dst_y1, dst_z0:dst_z1]   # view into vol
                    dst[...] = piece

                    # Erase implausible label IDs as each region lands.  Cheap (one pass
                    # over a block-sized view) and it keeps a phantom "label 18024" out of
                    # the inventory.  A sentinel reads as "not this label" whether erased
                    # or left in place, so this neither creates nor closes a hole in any
                    # surface.
                    bad = dst > _MAX_PLAUSIBLE_LABEL
                    n_bad = int(bad.sum())
                    if n_bad:
                        dst[bad] = 0
                        n_sentinel += n_bad
                n_placed += 1
                per_level[lv] = per_level.get(lv, 0) + 1

        levels_txt = ", ".join(f"L{lv}:{n}" for lv, n in sorted(per_level.items()))
        _log(f".syk: placed {n_placed} block(s) — {levels_txt}"
             f"  (deepest level {max_level}, {n_grid ** 3} slots)")
        if any(lv < max_level for lv in per_level):
            coarse = sum(n for lv, n in per_level.items() if lv < max_level)
            _log(f".syk: adaptive octree — {coarse} block(s) sit above the deepest "
                 f"level and were upsampled into place, {n_partial} of them only in the "
                 f"octants with no stored child")
        if placeholder:
            _log(f".syk: skipped {len(placeholder)} empty placeholder block(s); their "
                 f"regions come from the nearest non-empty ancestor")
        if n_sentinel:
            _log(f".syk: erased {n_sentinel} sentinel voxel(s) with IDs > "
                 f"{_MAX_PLAUSIBLE_LABEL} before meshing")

        # No seam-filling pass here on purpose — the block trim fix should have fixed the
        # gap issue, and patching gaps afterward costs ~20 s and tens of GB in temporaries.
        # If a file does show gaps at block boundaries, adjust the trim in the options
        # menu and check with the seam probe below (diagnostics; a clean join reads
        # '1111|1111').
        if diagnostics:
            _probe_seams(vol, ax, ay, az, (bbox_x0, bbox_y0, bbox_z0))

        clip_info = (bbox_x0, bbox_y0, bbox_z0, full_nx, full_ny, full_nz)
        return vol, clip_info

    except PermissionError as exc:
        _log(f".syk locked (sharing violation): {exc}")
        raise _SykLockedError(syk_path) from exc
    except Exception:
        _log(f".syk parse failed:\n{traceback.format_exc()}")
        return None, None


def _scan_block_headers(f, n_blocks, block_stride: int, block_payload: int) -> dict:
    """
    Walk the fixed-stride block records and return {block_id: file offset}.

    Only headers are read, never payloads, so this is cheap even on a multi-GB file.
    Records whose declared payload size disagrees with the header geometry are skipped,
    and the walk stops at the first record without the 'fvgu' magic — which is also how
    the trailing INDX footer terminates the scan.
    """
    block_offset = {}
    off = 36
    for _ in range(n_blocks):
        f.seek(off)
        hdr = f.read(24)
        if len(hdr) < 24 or hdr[0:4] != _FVGU:
            break
        pl_size, bid, _lod, _flags = struct.unpack("<QIII", hdr[4:])
        if pl_size == block_payload:
            block_offset[bid] = off
        off += block_stride
    return block_offset


def _estimate_bbox(f, block_offset: dict, block_dims, block_payload: int,
                   n_grid: int, full_grid):
    """
    Bounding box of the painted region, in finest-grid voxels, from the root block alone.

    The root is a whole-volume preview, so a single small read tells us which part of the
    (potentially billions of voxels) full grid is worth allocating.  The whole block is
    used, overlap included: that can only widen the estimate, and a generous bbox is
    harmless.  Falls back to the full grid when there is no root block.

    Returns ((x0, y0, z0), (x1, y1, z1)).
    """
    cx, cy, cz = block_dims
    full_nx, full_ny, full_nz = full_grid
    if 0 not in block_offset:
        return (0, 0, 0), (full_nx, full_ny, full_nz)

    f.seek(block_offset[0] + 24)
    root = (np.frombuffer(f.read(block_payload), dtype="<u2")
            .reshape(cz, cy, cx).transpose(2, 1, 0))
    nz = np.where(root > 0)
    if len(nz[0]) == 0:
        return (0, 0, 0), (full_nx, full_ny, full_nz)

    m = _ROOT_BBOX_MARGIN_BLOCKS
    lo, hi = [], []
    for idx, full in zip(nz, (full_nx, full_ny, full_nz)):
        lo.append(max(0, (int(idx.min()) - m) * n_grid))
        hi.append(min(full, (int(idx.max()) + m + 1) * n_grid))
    return tuple(lo), tuple(hi)


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
# Per-label field construction
# -----------------------------------------------------------------------

def _padded_span(flags: np.ndarray, n: int, margin: int):
    """First..last True index of `flags`, padded by `margin`, clamped to [0, n)."""
    idx = np.nonzero(flags)[0]
    return max(0, int(idx[0]) - margin), min(n, int(idx[-1]) + 1 + margin)


def _mask_bbox(mask: np.ndarray, margin: int):
    """
    Bounding box of the True voxels, padded by `margin` and clamped to the array.
    Returns ((ox, oy, oz), (x1, y1, z1)) with exclusive upper bounds, or None if empty.
    """
    xs = np.any(mask, axis=(1, 2))
    if not xs.any():
        return None
    ys = np.any(mask, axis=(0, 2))
    zs = np.any(mask, axis=(0, 1))
    ox, x1 = _padded_span(xs, mask.shape[0], margin)
    oy, y1 = _padded_span(ys, mask.shape[1], margin)
    oz, z1 = _padded_span(zs, mask.shape[2], margin)
    return (ox, oy, oz), (x1, y1, z1)


def _label_bbox(label_vol: np.ndarray, label_id: int, margin: int):
    """
    Bounding box of one label in the full volume — same contract as _mask_bbox, but
    scanned in X slabs so no full-volume boolean is materialised.  label_vol can be
    billions of voxels, and prep for one label runs on a worker thread while the
    previous label uploads, so two full-volume temporaries could coexist.
    """
    nx, ny, nz = label_vol.shape
    xs = np.zeros(nx, dtype=bool)
    ys = np.zeros(ny, dtype=bool)
    zs = np.zeros(nz, dtype=bool)
    slab = 64
    for x0 in range(0, nx, slab):
        eq = label_vol[x0:x0 + slab] == label_id
        xs[x0:x0 + eq.shape[0]] = eq.any(axis=(1, 2))
        ys |= eq.any(axis=(0, 2))
        zs |= eq.any(axis=(0, 1))
    if not xs.any():
        return None
    ox, x1 = _padded_span(xs, nx, margin)
    oy, y1 = _padded_span(ys, ny, margin)
    oz, z1 = _padded_span(zs, nz, margin)
    return (ox, oy, oz), (x1, y1, z1)


def _label_components(mask: np.ndarray):
    """
    6-connected component labeling of a boolean volume.

    Returns (comp, n): comp is int32 with 0 = background and 1..n numbering the
    components.  Uses scipy.ndimage.label when available; otherwise the pure-NumPy
    fallback below (same contract, so the two are interchangeable).
    """
    if _scipy_label is None:
        return _label_components_numpy(mask)
    comp, n = _scipy_label(mask)
    return comp.astype(np.int32, copy=False), int(n)


def _label_components_numpy(mask: np.ndarray):
    """
    scipy-free 6-connected component labeling: union-find over z-runs.

    Runs along Z (the contiguous axis) are the unit of work instead of voxels — neuron
    components are elongated, so run counts stay far below voxel counts and a
    frontier-BFS's repeated full-volume dilations are avoided.  Runs in z-adjacent
    overlap within a row are the same run by construction; runs in Y- and X-neighbour
    rows are unioned when their z-intervals overlap.
    """
    m = np.ascontiguousarray(mask, dtype=bool)
    nx, ny, nz = m.shape
    n_rows = nx * ny
    pad = np.zeros((n_rows, nz + 2), dtype=np.int8)
    pad[:, 1:-1] = m.reshape(n_rows, nz)
    d = np.diff(pad, axis=1)
    run_row, run_z0 = np.nonzero(d == 1)     # row-major order: sorted by row, then z
    _, run_z1 = np.nonzero(d == -1)          # same rows in the same order, exclusive end
    n_runs = len(run_row)
    if n_runs == 0:
        return np.zeros(m.shape, np.int32), 0

    # runs of row r are indices row_start[r] .. row_start[r+1]
    row_start = np.searchsorted(run_row, np.arange(n_rows + 1))
    parent = np.arange(n_runs, dtype=np.int64)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]    # path halving
            i = parent[i]
        return i

    def merge_rows(r, q):
        """Union runs of rows r and q whose z-intervals overlap (two-pointer walk)."""
        i, i_end = row_start[r], row_start[r + 1]
        j, j_end = row_start[q], row_start[q + 1]
        while i < i_end and j < j_end:
            if run_z1[i] <= run_z0[j]:
                i += 1
            elif run_z1[j] <= run_z0[i]:
                j += 1
            else:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
                if run_z1[i] <= run_z1[j]:   # advance whichever interval ends first
                    i += 1
                else:
                    j += 1

    occupied = np.nonzero(row_start[1:] > row_start[:-1])[0]
    for r in occupied:
        if r % ny > 0 and row_start[r - 1] < row_start[r]:        # Y neighbour (x, y-1)
            merge_rows(r, r - 1)
        if r >= ny and row_start[r - ny] < row_start[r - ny + 1]:  # X neighbour (x-1, y)
            merge_rows(r, r - ny)

    roots = np.fromiter((find(i) for i in range(n_runs)), np.int64, n_runs)
    uniq, comp_of_run = np.unique(roots, return_inverse=True)
    out = np.zeros((n_rows, nz), np.int32)
    for i in range(n_runs):
        out[run_row[i], run_z0[i]:run_z1[i]] = comp_of_run[i] + 1
    return out.reshape(nx, ny, nz), len(uniq)


def _signed_field(sub: np.ndarray, sigma: float) -> np.ndarray:
    """
    Signed uint8 field for AddSurface from a boolean mask.  Imaris reads the uint8 as
    signed int8 and meshes the zero crossing:
      sigma <= 0 : binary 100 (inside) / 200 (outside) — voxel fidelity (blocky surface,
                   nothing eroded).
      sigma  > 0 : a Gaussian-blurred gradient (+inside … −outside, 0 at the boundary), so
                   Imaris' marching cubes interpolates a smooth triangulated surface; larger
                   sigma = smoother (and rounds off / drops fine detail).
    """
    if sigma > 0:
        if _scipy_gaussian_filter is None:
            blur = _smooth_mask_3d(sub, iterations=max(1, int(round(2 * sigma))))
        else:
            blur = _scipy_gaussian_filter(sub.astype(np.float32), sigma=sigma)
        signed = np.clip((blur - 0.5) * 200.0, -120, 120)     # +inside, −outside, 0 at surface
        return signed.astype(np.int8).view(np.uint8)
    return np.where(sub, 100, 200).astype(np.uint8)          # blocky, voxel fidelity


def _prep_label(label_vol, label_id, sigma=0.0, margin=_PREP_MARGIN_VOXELS,
                on_progress=None):
    """
    Build the signed uint8 fields AddSurface meshes for one label — one field per
    disconnected component, each cropped to its own bounding box (CPU-only, thread-safe —
    no COM/Ice access).

    Components are separated so every AddSurface call makes its own surface object in
    Imaris; that's what lets debris be selected and size-filtered there.  Each component
    is smoothed on its own, in isolation from the rest of the label: blurring the whole
    label at once would leak a nearby component's Gaussian tail into this component's
    crop and mesh phantom shells there.

    Returns None if the label is absent, else a dict:
      components:     [(field, (ox,oy,oz) offset in label_vol, comp_voxels, n_kept), …]
                      ordered largest component first
      n_components:   component count found in the label
      n_vanished / vanished_voxels: components the blur erased entirely (no voxel stayed
                      above 0.5, so there is no zero crossing to mesh) — skipped rather
                      than sent through a pointless AddSurface call
      n_painted:      voxels of this label in label_vol
    `margin` grows with sigma so the blur has room to fall back to "outside" in-crop.
    `on_progress(text, done, total)`, if given, is called as the work advances; a total
    of 0 means the step has no count.
    """
    m = max(margin, int(np.ceil(3 * sigma)) + 1)   # room for the Gaussian tail
    bbox = _label_bbox(label_vol, label_id, m)
    if bbox is None:
        return None
    (lx, ly, lz), (x1, y1, z1) = bbox

    sub = label_vol[lx:x1, ly:y1, lz:z1] == label_id   # crop-sized boolean only
    n_painted = int(sub.sum())

    if on_progress is not None:
        on_progress("Finding separate pieces", 0, 0)
    comps, n_comps = _label_components(sub)
    del sub
    sizes = np.bincount(comps.ravel())       # sizes[0] = background, ignored
    keep = sorted(range(1, n_comps + 1), key=lambda cid: -sizes[cid])

    components = []
    n_vanished = vanished_voxels = 0
    for piece_index, cid in enumerate(keep):
        if on_progress is not None:
            on_progress(f"Preparing piece {piece_index + 1} of {n_comps}",
                        piece_index, n_comps)
        cmask = comps == cid                 # isolation: neighbours excluded before blur
        (ox, oy, oz), (cx1, cy1, cz1) = _mask_bbox(cmask, m)
        field = _signed_field(cmask[ox:cx1, oy:cy1, oz:cz1], sigma)
        n_kept = int((field.view(np.int8) > 0).sum())
        if n_kept == 0:                      # blur erased it: nothing crosses zero
            n_vanished += 1
            vanished_voxels += int(sizes[cid])
            continue
        components.append((field, (lx + ox, ly + oy, lz + oz), int(sizes[cid]), n_kept))

    return {"components": components, "n_components": n_comps,
            "n_vanished": n_vanished, "vanished_voxels": vanished_voxels,
            "n_painted": n_painted}


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
    Split an (nx, ny, nz) uint8 field into upload bands for _upload_field.

    Returns a list of (z0, z1, y_strips); each (z-range, y-strip) pair is one COM call and
    stays under `budget` bytes.  Whole z-bands are used
    when a full XY slice fits in the budget; otherwise one z-slice per band, tiled in Y.

    `budget` defaults to _ICE_CHUNK_BYTES read at call time, not import time, so the
    constant stays adjustable.

    Bands are never split along X, so one X row (nx bytes for a uint8 field) is the finest
    granularity and a budget smaller than that cannot be honoured.  Reaching it would take
    a crop wider than _ICE_CHUNK_BYTES voxels; no such file has come up so far, so the case
    is left unhandled rather than guarded against.
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


def _upload_field(ds, field: np.ndarray, budget: int | None = None) -> int:
    """
    Upload a cropped (nx,ny,nz) uint8 field to an IDataSet via SetDataSubVolumeAs1DArrayBytes.

    Imaris' 1D layout is X-fastest, then Y, then Z.  field is (X,Y,Z), so transposing a
    sub-block to (Z,Y,X) and calling the default C-order .tobytes() yields x-fastest bytes.
    Returns the number of COM calls made.
    """
    nx, ny, nz = field.shape
    calls = 0
    for z0, z1, strips in _upload_bands(nx, ny, nz, budget):
        for y0, y1 in strips:
            blob = field[:, y0:y1, z0:z1].transpose(2, 1, 0).tobytes()
            ds.SetDataSubVolumeAs1DArrayBytes(blob, 0, y0, z0, 0, 0, nx, y1 - y0, z1 - z0)
            calls += 1
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


def _syk_is_locked(path: str) -> bool:
    """
    True if the .syk cannot be opened for reading — on Windows that is syGlass holding
    the file (sharing violation surfaces as PermissionError).  GUI-free so it is testable
    without a display.
    """
    try:
        with open(path, "rb") as f:
            f.read(1)
        return False
    except PermissionError:
        return True


def _ensure_syk_readable(path: str, ask_retry) -> bool:
    """
    Block until the .syk is readable or the user gives up.  Returns True when readable.

    While the file is locked, `ask_retry(message)` shows a Retry/Cancel prompt and
    returns True for Retry, so the user can close the project in syGlass without
    restarting the XTension.  The wording covers both lock states seen in the wild:
    syGlass sometimes keeps the .syk handle open after the project itself is closed, in
    which case only exiting syGlass releases it.
    """
    msg = (f"The .syk file is locked by syGlass:\n\n{path}\n\n"
           f"{_LOCKED_SYK_HINT}\n\nThen press Retry.")
    while _syk_is_locked(path):
        _log(f".syk locked by syGlass: {path}")
        if not ask_retry(msg):
            return False
    return True


def _ask_retry_dialog(message: str) -> bool:
    """Standalone Retry/Cancel dialog, for use before the import window exists."""
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    retry = messagebox.askretrycancel("Import from syGlass", message, parent=root)
    root.destroy()
    return retry


def _ask_for_syk(initialdir: str) -> str | None:
    """
    Open a modal file dialog for the .syk, starting in the open .ims's own directory.

    Doesn't remember the last folder used — in a core setting, best not to carry over
    the previous user's folder when the current .ims already points to a better guess.
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
    ("Light", 1.0),
    ("Medium", 1.5),
    ("Strong", 2.5),
]

# The preset selected each time the dialog opens.  Must be one of the _SMOOTHING_LEVELS
# values or the radio buttons won't pre-select it.
_DEFAULT_SIGMA = 1.0


@dataclass
class _Settings:
    """Choices made on the options page."""
    sigma: float
    diagnostics: bool
    trim: tuple[int, int, int]


# -----------------------------------------------------------------------
# Progress reporting
# -----------------------------------------------------------------------

@dataclass
class _ImportStatus:
    """
    Progress of the import, written by the import and prep threads and read by the
    window thread, which redraws from it several times a second.  Fields are set one
    at a time without a lock, so a redraw can catch an update halfway; the next redraw
    shows it complete.
    """
    stage: str = ""                  # "read", "labels", "build" or "done"
    stage_start: float = 0.0
    stage_durations: dict = dataclass_field(default_factory=dict)

    # The line under the stage list and the bar beneath it.  A total of 0 means the
    # current step has no count, and the bar animates instead.
    activity: str = "Starting"
    activity_done: int = 0
    activity_total: int = 0

    # Inputs to the finish-time estimate, set during the build stage.
    n_labels: int = 0
    label_voxels: list = dataclass_field(default_factory=list)
    label_index: int = 0
    waiting_for_prep: bool = False
    pieces_in_label: int = 0
    pieces_built: int = 0
    voxels_built_in_label: int = 0

    def begin_stage(self, stage: str) -> None:
        now = time.time()
        if self.stage:
            self.stage_durations[self.stage] = now - self.stage_start
        self.stage = stage
        self.stage_start = now

    def set_activity(self, text: str, done: int = 0, total: int = 0) -> None:
        self.activity_done = done
        self.activity_total = total
        self.activity = text

    def begin_label(self, label_index: int) -> None:
        self.pieces_in_label = 0
        self.pieces_built = 0
        self.voxels_built_in_label = 0
        self.label_index = label_index

    def piece_built(self, voxels: int) -> None:
        self.pieces_built += 1
        self.voxels_built_in_label += voxels

    def build_fractions(self) -> tuple[float, float]:
        """Share of the build that is done, measured by voxels and by pieces."""
        total_voxels = sum(self.label_voxels)
        finished_voxels = sum(self.label_voxels[:self.label_index])
        voxel_fraction = (finished_voxels + self.voxels_built_in_label) / total_voxels
        if self.pieces_in_label > 0:
            label_part = self.pieces_built / self.pieces_in_label
        else:
            label_part = 0.0
        piece_fraction = (self.label_index + label_part) / self.n_labels
        return voxel_fraction, piece_fraction


def _estimate_remaining(elapsed: float, voxel_fraction: float,
                        piece_fraction: float) -> float | None:
    """
    Seconds of build left, projected from the time spent and the share of the build done.

    The share is the smaller of the voxel and piece measures, so whichever is further
    behind sets the estimate.  The projection is then padded by _ETA_PESSIMISM, scaled
    down as the build completes.  Returns None until some of the build is done.
    """
    fraction = min(voxel_fraction, piece_fraction)
    if fraction <= 0.0:
        return None
    if fraction >= 1.0:
        return 0.0
    projected = elapsed / fraction - elapsed
    return projected * (1.0 + _ETA_PESSIMISM * (1.0 - fraction))


def _format_duration(seconds: float) -> str:
    """Seconds as m:ss, or h:mm:ss from an hour up."""
    minutes, secs = divmod(int(max(0.0, seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


@dataclass
class _Summary:
    """Contents of the summary page shown when the import ends."""
    succeeded: bool
    headline: str
    details: list[str]
    warnings: list[str]
    log_path: str | None


class _Window:
    """
    The import's one window: the options page, then progress, then a summary that stays
    open until the user closes it.

    Tk runs in its own thread, so the window keeps redrawing while the import thread is
    busy in NumPy or waiting on Imaris.  The import thread never calls Tk itself.  It
    queues functions on `_requests`, which the Tk thread runs, and waits on `_replies`
    for the user's answers.
    """

    _TITLE = "Import from syGlass"
    _WIDTH = 560
    _POLL_MS = 100
    _BAR_STEPS = 1000
    _STAGES = [("read", "Read mask from .syk"),
               ("labels", "Find labels"),
               ("build", "Build surfaces in Imaris")]

    def __init__(self) -> None:
        self._requests = queue.Queue()
        self._replies = queue.Queue()
        self._closed = threading.Event()
        self._startup_error = None
        started = threading.Event()
        self._thread = threading.Thread(target=self._run_tk, args=(started,), daemon=True)
        self._thread.start()
        started.wait()
        if self._startup_error is not None:
            raise RuntimeError("could not open the import window") from self._startup_error

    # ---- Called from the import thread ----------------------------------

    def ask_options(self) -> _Settings | None:
        """Show the options page and wait for Run (settings) or Cancel (None)."""
        self._requests.put(self._show_options)
        return self._wait_for_reply()

    def ask_retry(self, message: str) -> bool:
        """Show a Retry/Cancel prompt over the window; True for Retry."""
        self._requests.put(lambda: self._show_retry(message))
        return self._wait_for_reply()

    def show_progress(self, status: _ImportStatus) -> None:
        self._requests.put(lambda: self._show_progress(status))

    def show_summary(self, summary: _Summary) -> None:
        """Show the summary page and wait until the user closes the window."""
        self._requests.put(lambda: self._show_summary(summary))
        self._closed.wait()

    def close(self) -> None:
        self._requests.put(self._root.quit)
        self._closed.wait()

    def _wait_for_reply(self):
        reply = self._replies.get()
        if isinstance(reply, Exception):
            raise reply
        return reply

    # ---- Tk thread ------------------------------------------------------

    def _run_tk(self, started: threading.Event) -> None:
        try:
            import tkinter as tk
            from tkinter import messagebox, ttk
            self._tk = tk
            self._ttk = ttk
            self._messagebox = messagebox
            self._root = tk.Tk()
            self._root.title(self._TITLE)
            self._root.resizable(False, False)
            self._root.protocol("WM_DELETE_WINDOW", self._on_close_button)
            self._root.report_callback_exception = self._on_tk_error
        except Exception as exc:
            self._startup_error = exc
            started.set()
            return
        self._page = None
        self._close_action = None
        self._positioned = False
        self._status = None
        self._progress = None
        started.set()
        self._root.after(self._POLL_MS, self._poll)
        self._root.mainloop()
        self._root.destroy()
        self._closed.set()

    def _on_tk_error(self, exc_type, exc, tb) -> None:
        """A bug in the window code: log it, then end the window so nothing waits forever."""
        _log("import window failed:\n" + "".join(traceback.format_exception(exc_type, exc, tb)))
        self._replies.put(RuntimeError("the import window failed; see the log"))
        self._root.quit()

    def _poll(self) -> None:
        while True:
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                break
            request()
        if self._status is not None:
            self._redraw_progress()
        self._root.after(self._POLL_MS, self._poll)

    def _on_close_button(self) -> None:
        # The progress page has no close action, so the window can't be closed mid-import.
        if self._close_action is not None:
            self._close_action()

    def _set_page(self, close_action, topmost: bool):
        """Replace the current page with an empty one and return it for filling."""
        if self._page is not None:
            self._page.destroy()
        self._page = self._tk.Frame(self._root, padx=16, pady=12)
        self._page.pack(fill="both", expand=True)
        self._close_action = close_action
        self._root.attributes("-topmost", topmost)
        return self._page

    def _fit(self) -> None:
        """Size the window to its page, centring it on the screen the first time."""
        self._root.update_idletasks()
        width = max(self._WIDTH, self._root.winfo_reqwidth())
        height = self._root.winfo_reqheight()
        if self._positioned:
            self._root.geometry(f"{width}x{height}")
        else:
            x = (self._root.winfo_screenwidth() - width) // 2
            y = (self._root.winfo_screenheight() - height) // 3
            self._root.geometry(f"{width}x{height}+{x}+{y}")
            self._positioned = True
        self._root.lift()
        self._root.focus_force()

    def _show_retry(self, message: str) -> None:
        retry = self._messagebox.askretrycancel(self._TITLE, message, parent=self._root)
        self._replies.put(retry)

    # ---- Options page ---------------------------------------------------

    def _show_options(self) -> None:
        """
        The options page always opens with the same defaults rather than remembering the
        last run — same reasoning as _ask_for_syk: in a core setting, best not to carry
        over the previous user's settings, especially a hand-edited block trim.
        """
        tk = self._tk
        page = self._set_page(close_action=self._cancel_options, topmost=True)

        # Two tabs.  Everything a normal import needs is on the first; the second holds
        # settings that are either diagnostic or not trustworthy yet, kept out of the
        # way so people aren't tempted to change something usually best left alone.
        tabs = self._ttk.Notebook(page)
        basic = tk.Frame(tabs)
        advanced = tk.Frame(tabs)
        tabs.add(basic, text="  Import  ")
        tabs.add(advanced, text="  Advanced  ")
        tabs.pack(fill="both", expand=True)

        # ---- Import tab: surface smoothing only ----------------------------
        tk.Label(basic, text="Surface smoothing", font=("", 10, "bold"),
                 anchor="w").pack(fill="x", padx=14, pady=(12, 0))
        tk.Label(basic, text="Gaussian blur applied before meshing. Smoother surfaces show "
                            "less voxel detail.",
                 anchor="w", justify="left", wraplength=430).pack(fill="x", padx=14,
                                                                  pady=(0, 6))
        sigma_var = tk.DoubleVar(value=_DEFAULT_SIGMA)

        crow = tk.Frame(basic)
        custom = tk.Entry(crow, width=8)

        def clear_custom():
            """A preset click wins over a stale custom value (see run)."""
            custom.delete(0, "end")

        # Presets show their blur value, so a user who wants something in between can
        # just type it in the custom box.
        for name, val in _SMOOTHING_LEVELS:
            text = name if val <= 0 else f"{name} (blur {val:g})"
            tk.Radiobutton(basic, text=text, variable=sigma_var, value=val, anchor="w",
                           command=clear_custom).pack(fill="x", padx=26)

        crow.pack(fill="x", padx=26, pady=(4, 12))
        tk.Label(crow, text="…or custom blur:").pack(side="left")
        custom.pack(side="left", padx=6)

        # No debris filter at import time: every disconnected piece becomes its own
        # surface object, so users size-filter interactively in Imaris (Filter tab)
        # instead of guessing a voxel threshold here.
        tk.Label(basic, text="Each disconnected piece becomes its own object in Imaris; "
                            "delete debris there via the Filter tab (e.g. \"Number of "
                            "Voxels\").",
                 anchor="w", justify="left", wraplength=430, fg="grey").pack(
                     fill="x", padx=14, pady=(0, 12))

        # ---- Advanced tab: diagnostics, block geometry ---------------------
        diag_var = tk.BooleanVar(value=False)
        tk.Checkbutton(advanced, text="Troubleshooting: log block diagnostics + mask preview",
                       variable=diag_var, anchor="w").pack(fill="x", padx=14, pady=(12, 10))

        tk.Label(advanced, text="Block trim X,Y,Z", font=("", 10, "bold"),
                 anchor="w").pack(fill="x", padx=14, pady=(6, 0))
        tk.Label(advanced,
                 text="Voxels dropped from each block edge, because blocks overlap their "
                      "neighbours. Only change this to diagnose a repeating artifact at "
                      "block boundaries.",
                 anchor="w", justify="left", wraplength=430).pack(fill="x", padx=14,
                                                                  pady=(0, 6))
        trow = tk.Frame(advanced)
        trow.pack(fill="x", padx=26, pady=(0, 12))
        trim_entry = tk.Entry(trow, width=12)
        trim_entry.insert(0, ",".join(str(v) for v in _DEFAULT_TRIM))
        trim_entry.pack(side="left")
        tk.Label(trow, text=f"  (default {','.join(str(v) for v in _DEFAULT_TRIM)})",
                 fg="grey").pack(side="left")

        def run():
            # Invalid input shows an error and leaves the page open for correction.
            # Reverting to a default would let the user believe they tested a value
            # they did not test.
            sigma = float(sigma_var.get())
            custom_text = custom.get().strip()
            if custom_text:
                try:
                    sigma = float(custom_text)
                except ValueError:
                    self._messagebox.showerror(
                        self._TITLE, f"Custom blur must be a number (got {custom_text!r}).",
                        parent=self._root)
                    return
            try:
                trim = tuple(int(v) for v in trim_entry.get().replace(" ", "").split(","))
                if len(trim) != 3:
                    raise ValueError
            except ValueError:
                self._messagebox.showerror(
                    self._TITLE,
                    "Block trim must be three integers separated by commas, e.g. 3,1,1.",
                    parent=self._root)
                return
            self._replies.put(_Settings(sigma=max(0.0, sigma),
                                        diagnostics=bool(diag_var.get()),
                                        trim=trim))
            # Switch pages at once so Run can't be pressed twice; the import thread
            # hands over its status object shortly after.
            self._show_progress(None)

        buttons = tk.Frame(page)
        buttons.pack(pady=(12, 0))
        tk.Button(buttons, text="Run", width=12, command=run).pack(side="left", padx=6)
        tk.Button(buttons, text="Cancel", width=12,
                  command=self._cancel_options).pack(side="left", padx=6)
        self._fit()

    def _cancel_options(self) -> None:
        self._close_action = None
        self._replies.put(None)

    # ---- Progress page --------------------------------------------------

    def _show_progress(self, status: _ImportStatus | None) -> None:
        """Build the progress page if it isn't showing yet, then redraw from `status`."""
        if self._progress is None:
            tk = self._tk
            page = self._set_page(close_action=None, topmost=False)
            stage_rows = {}
            stage_frame = tk.Frame(page)
            stage_frame.pack(fill="x")
            for row, (key, name) in enumerate(self._STAGES):
                name_label = tk.Label(stage_frame, text=name, anchor="w")
                name_label.grid(row=row, column=0, sticky="w", pady=1)
                state_label = tk.Label(stage_frame, text="", anchor="e", fg="grey")
                state_label.grid(row=row, column=1, sticky="e", pady=1)
                stage_rows[key] = (name_label, state_label)
            stage_frame.columnconfigure(0, weight=1)

            activity = tk.Label(page, text="Starting", anchor="w", justify="left",
                                wraplength=self._WIDTH - 40)
            activity.pack(fill="x", pady=(12, 4))
            bar = self._ttk.Progressbar(page, length=self._WIDTH - 32, mode="determinate",
                                        maximum=self._BAR_STEPS)
            bar.pack(fill="x")
            timing = tk.Label(page, text="", anchor="w", fg="grey")
            timing.pack(fill="x", pady=(6, 0))
            self._progress = {"start": time.time(), "stage_rows": stage_rows,
                              "activity": activity, "bar": bar, "bar_mode": "determinate",
                              "timing": timing}
            self._fit()
        if status is not None:
            self._status = status

    def _redraw_progress(self) -> None:
        status = self._status
        widgets = self._progress
        now = time.time()

        for key, (name_label, state_label) in widgets["stage_rows"].items():
            if key in status.stage_durations:
                name_label.config(font=("", 9))
                state_label.config(text=f"done in {_format_duration(status.stage_durations[key])}")
            elif key == status.stage:
                name_label.config(font=("", 9, "bold"))
                state_label.config(text=_format_duration(now - status.stage_start))
            else:
                name_label.config(font=("", 9))
                state_label.config(text="")

        widgets["activity"].config(text=status.activity)

        bar = widgets["bar"]
        if status.activity_total > 0:
            if widgets["bar_mode"] != "determinate":
                bar.stop()
                bar.config(mode="determinate")
                widgets["bar_mode"] = "determinate"
            bar.config(value=self._BAR_STEPS * status.activity_done / status.activity_total)
        elif widgets["bar_mode"] != "indeterminate":
            bar.config(mode="indeterminate", value=0)
            bar.start(20)
            widgets["bar_mode"] = "indeterminate"

        elapsed_text = f"Elapsed {_format_duration(now - widgets['start'])}"
        if status.stage == "build":
            remaining = _estimate_remaining(now - status.stage_start, *status.build_fractions())
            if remaining is None:
                finish_text = "estimating finish time…"
            else:
                finish_clock = datetime.datetime.fromtimestamp(now + remaining)
                finish_text = f"estimated finish around {finish_clock.strftime('%H:%M')}"
        else:
            finish_text = "finish time is estimated once surfaces start building"
        widgets["timing"].config(text=f"{elapsed_text}   ·   {finish_text}")

    # ---- Summary page ---------------------------------------------------

    def _show_summary(self, summary: _Summary) -> None:
        tk = self._tk
        self._status = None
        self._progress = None
        page = self._set_page(close_action=self._root.quit, topmost=False)
        wrap = self._WIDTH - 40

        tk.Label(page, text=summary.headline, font=("", 11, "bold"),
                 anchor="w").pack(fill="x")
        for line in summary.details:
            tk.Label(page, text=line, anchor="w", justify="left",
                     wraplength=wrap).pack(fill="x", pady=(6, 0))

        if summary.warnings:
            tk.Label(page, text="Warnings", font=("", 10, "bold"),
                     anchor="w").pack(fill="x", pady=(12, 0))
            for warning in summary.warnings:
                tk.Label(page, text=f"• {warning}", anchor="w", justify="left",
                         wraplength=wrap).pack(fill="x", pady=(2, 0))

        if summary.log_path:
            tk.Label(page, text=f"Log: {summary.log_path}", anchor="w", justify="left",
                     wraplength=wrap, fg="grey").pack(fill="x", pady=(12, 0))

        buttons = tk.Frame(page)
        buttons.pack(pady=(12, 0))
        if summary.log_path and sys.platform == "win32":
            tk.Button(buttons, text="Open log", width=12,
                      command=lambda: os.startfile(summary.log_path)).pack(side="left", padx=6)
        tk.Button(buttons, text="Close", width=12,
                  command=self._root.quit).pack(side="left", padx=6)
        self._fit()


# -----------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------

def _warn_dialog(message: str) -> None:
    """
    Show a modal warning dialog and log the same message.  Use this instead of _warn
    whenever the user needs to see it without having to open the log.
    """
    _warn(message)
    try:
        import tkinter.messagebox
        tkinter.messagebox.showwarning("Import from syGlass", message)
    except Exception:
        pass
