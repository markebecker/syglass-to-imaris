# SyGlass Segmentations ↔ Imaris

Imaris XTension to convert syGlass masks to Imaris surfaces.

## Installation:
1. In Imaris, navigate to CustomTools Preferences.
2. Ensure that a valid python.exe is selected. For Imaris 11, it must be Python 3.11; for Imaris 10, Python 3.7.
3. Install the plugin: Download `XT_import_from_syglass.py` and copy it to a CustomTools directory.  Alternatively, `git pull` this repository to the path of your choice and add that path to the list of CustomTools Python XTension directories.

### Dependencies:
* `numpy` — required.
* `scipy` — optional. Used for surface smoothing and connected-component labeling; without it built-in fallbacks are used instead.

This XTension has been tested on Windows 11 only, with the following Imaris and syGlass versions:
* Imaris 10.2.0
* Imaris 11.0.1
* syGlass v2.6.0

Performance with other versions or on other platforms may vary.

## Usage:
1. Open the `.ims` in Imaris and run Extensions → Import from syGlass.
2. The matching `.syk` is found automatically if it sits beside the `.ims` and shares its name; otherwise a file picker opens.
3. Pick a surface smoothing level. "None" gives exact voxel fidelity (blocky surface); higher settings interpolate a smoother mesh at the cost of fine detail.
4. Each label becomes one Surfaces item in the scene tree, and every disconnected piece of that label is its own surface object inside it — so small debris can be selected and deleted in Imaris afterwards (Filter tab, e.g. "Number of Voxels").
5. Surfaces are added to the Surpass scene but not saved — press Ctrl+S in Imaris when you are happy with them.

**Troubleshooting:**
* If syGlass holds the `.syk` open, the import may show a Retry/Cancel dialog rather than failing. Close the project in syGlass and press Retry; if it still fails, exit syGlass entirely.
* Every run writes a timestamped log to a `logs/` folder next to the script. Check it if something seems off.
* The options menu has a troubleshooting toggle that adds block-geometry diagnostics, a seam probe across block boundaries, and ASCII previews of the reconstructed mask to that log. These are independent of Imaris, so they distinguish a bad mask read (likely a corrupted or wrong `.syk`) from a bad surface render.

**Known limitations:**
* Single timepoint. Masks are imported onto t=0, so only the first timepoint of a time series receives surfaces.
* Block geometry within the `.syk` was reverse-engineered. The XTension re-measures it on every import and writes a warning to the log if the file disagrees — if you see one, check the resulting surfaces carefully before using them.


## Future Development:
* Potentially, at some point, a standalone app to port Imaris surfaces (or other surfaces...) into syGlass projects as masks.

License: [GPL-3.0](LICENSE)
