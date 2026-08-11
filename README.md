# Imaris ↔ syGlass Interoperability Pipeline

Goal is shuttle annotations/segmentations between `.ims` files and syGlass projects in user-friendly ways.

Currently:
* syGlass -> Imaris up and running for masks
* Imaris -> syGlass works natively for points (counting points in syGlass)

Future Development:
* Testing syGlass -> Imaris counting points
* syGlass plugin to port Imaris surfaces (or other surfaces...) into syGlass projects.
  * Imaris surface decoding to binary works, but directly making ROI(s) and mask objects via syGlass API has been problematic, particularly for Imaris-based projects (ie .ims + .syk)

Installation:
1. In Imaris 11.0.1, navigate to CustomTools Preferences.
2. Ensure that a valid Python 3.11 python.exe is selected.
3. Install the plugin (simple method): Download `XT_import_from_syglass.py` and copy it to a CustomTools directory. 
* You may also `git pull` this repository to the path of your choice and add that path to the list of CustomTools Python XTension directories. This method requires slightly more technical knowledge, but allows you to quickly easily update in the event of changes.

Dependencies:
* `numpy` — required.
* `scipy` — optional. Used for surface smoothing and connected-component labeling; without it built-in fallbacks are used instead.
* `leveldb` — optional, and only for the experimental counting-point import. Not needed for masks.

Usage:
1. Open the `.ims` in Imaris and run Extensions → Import from syGlass.
2. The matching `.syk` is found automatically if it sits beside the `.ims` and shares its name; otherwise a file picker opens.
3. Pick a surface smoothing level. "None" gives exact voxel fidelity (blocky); higher settings interpolate a smoother mesh at the cost of fine detail. The options open with fixed defaults every run — nothing is remembered from the previous user.
4. Each label becomes one Surfaces item in the scene tree, and every disconnected piece of that label is its own surface object inside it — so small debris can be selected and deleted in Imaris afterwards (Filter tab, e.g. "Number of Voxels"). Alternatively, set "Minimum object size" in the options to drop debris below that many voxels at import.
5. Surfaces are added to the Surpass scene but **not saved** — press Ctrl+S in Imaris when you are happy with them.

Known limitations:
* **Single timepoint.** Masks are imported onto t=0, so only the first timepoint of a time series receives surfaces. Multi-channel files are fine.
* **Counting-point import is experimental** and off by default. The `.sym` record layout and the syGlass → Imaris coordinate transform are both unconfirmed, so imported point positions should not be trusted. The checkbox exists to help work the format out.
* Block geometry within the `.syk` was reverse-engineered from real files. The XTension re-measures it on every import and writes a warning to the log if the file disagrees — if you see one, check the resulting surfaces carefully before using them.

Troubleshooting:
* If syGlass holds the `.syk` open, the import shows a Retry/Cancel dialog rather than failing. Close the project in syGlass and press Retry; if it still fails, exit syGlass entirely — syGlass can keep the file handle open even after the project is closed.
* Every run writes a timestamped log to a `logs/` folder next to the script; check it first.
* The options menu has a troubleshooting toggle that adds block-geometry diagnostics, a seam probe across block boundaries, and ASCII previews of the reconstructed mask to that log. These are independent of Imaris, so they distinguish a bad mask read from a bad surface render.

This XTension has been tested on Windows 11 only, with the following Imaris and syGlass versions:
* Imaris 10.2.0
* Imaris 11.0.1
* syGlass v2.6.0
Performance with other versions or on other platforms may vary.
