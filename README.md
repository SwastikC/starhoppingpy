# starhoppingpy

**A Python pipeline for star-hopping reference differential imaging (RDI) with VLT/SPHERE-IRDIS.**

`starhoppingpy` takes raw SPHERE-IRDIS frames all the way to a PSF-subtracted image. It sorts the raw data, builds calibrations, finds the star center, does the basic reduction, and runs LOCI-based reference-star subtraction. It is a Python port of the original SPHERE star-hopping reduction scripts (IDL), and keeps the same file conventions and processing steps.

---

## What is star-hopping?

High-contrast imaging has to remove the bright stellar PSF to reveal faint companions and circumstellar disks. Angular differential imaging (ADI) uses field rotation for this, but it struggles close to the star and with extended or face-on disks, where the field rotation also wipes out the astrophysical signal.

**Star-hopping** ([Wahhaj et al. 2021](https://ui.adsabs.harvard.edu/abs/2021A%26A...648A..26W)) is a form of RDI built for SPHERE. The telescope switches back and forth between the science target and a nearby reference star of similar brightness and color within a single observation, with only about a minute of overhead per hop. Because both stars are observed under nearly the same conditions and AO correction, the reference PSFs match the science PSFs closely. The stellar PSF can then be subtracted without depending on field rotation.

This package does the data handling and subtraction for such sequences.

---

## Features

- **Raw-data sorting.** Groups `SPH*.fits` files by observation container and sorts science, flat, sky, dark, center, flux and IFS calibration frames into per-night folders with ready-made file lists.
- **Calibrations.** Builds IRDIS flats and bad-pixel maps, skies, and darks, and cleans cosmic rays and hot pixels (`fixpix`, `qzap`).
- **Star centering.** Finds the star center automatically from the four satellite spots in `OBJECT,CENTER` frames, separately for the left and right IRDIS channels.
- **Basic reduction.** Applies calibrations frame by frame, normalizes flux, and builds de-rotation-ready ASDI cubes with parallactic angles stored alongside.
- **Frame selection.** Rejects bad frames based on residual RMS in chosen annuli.
- **PSF subtraction.** Rescales the reference PSF, ranks reference frames for each science frame, and runs region-restricted LOCI with iterative signal and background masking.
- **Utilities.** FITS I/O (including compressed `.fits.Z`), quick-look display helpers, radial profiles, RMS and SNR maps, robust statistics, and photometry helpers.

---

## Installation

Requires **Python ≥ 3.9**.

```bash
git clone https://github.com/SwastikC/starhoppingpy.git
cd starhoppingpy
pip install -e .
```

Optional extras:

```bash
pip install -e ".[extras]"     # unlzw3 (.fits.Z), scikit-image, fpdf
pip install -e ".[notebooks]"  # jupyter, ipywidgets
```

| Dependency | Needed for |
| --- | --- |
| `numpy`, `scipy`, `astropy`, `matplotlib` | Core pipeline (always installed) |
| `unlzw3` | Reading compressed ESO archive files (`*.fits.Z`) |
| `scikit-image` | Image resizing in `basic.py` |
| `fpdf` | `basic.pdf_creator` |
| `pysynphot` | `zfuncs.zp` photometry (also needs the `PYSYN_CDBS` reference files). Not included in the extras; install it yourself if needed. |

### External tools

Sorting raw data (`sh_sort`) calls the ESO **`dfits`** and **`fitsort`** command-line tools to read FITS headers. They ship with the ESO pipeline tools / `eclipse` and must be on your `PATH`:

```bash
which dfits fitsort
```

---

## Package layout

```
starhoppingpy/
├── pyproject.toml
├── src/starhoppingpy/
│   ├── sh_sort.py      # sort raw SPH*.fits into science/calibration sets
│   ├── sh_funcs.py     # calibrations, centering, basic reduction, ASDI cubes, ref-PSF rescaling
│   ├── basic.py        # centroiding, fixpix, robust sigma, radial profiles, masks
│   ├── functions.py    # LOCI / RDI subtraction and reference selection
│   └── zfuncs.py       # FITS I/O, display, photometry and general helpers
└── notebooks/
    ├── sphere_sort_hopsets.ipynb
    ├── irdis_basic_reduc_starhop.ipynb
    └── double_loci_restrict3_npz.ipynb
```

| Module | Key functions |
| --- | --- |
| `sh_sort` | `get_file_list`, `parse_dfits_line`, `group_datasets`, `process_science_files`, `process_calibration_{flat,sky,dark,center,flux,ifscal}` |
| `sh_funcs` | `irdis_makeflat`, `irdis_make_sky`, `irdis_badpix_from_obj2`, `sph_cen_irdis`, `irdis_make_flux_cube`, `irdis_basicreduc`, `irdis_make_asdi_cube`, `sph_remove_bad_images`, `sph_rescale_ref_psf` |
| `basic` | `bscentrd`, `fixpix`, `get_mask`, `robust_sigma`, `radial_profile`, `snmap`, `rmsmap` |
| `functions` | `remove_bgnd_slope`, `loci_operation`, `good_refs_mask`, `best_refs`, `zscalediff2`, `rgn_fac` |
| `zfuncs` | `readfits`, `writefits`, `zdisp`/`zdisp2`/`zdisp3`, `getfreq`, `rmsmap_ann`, `dist_circle`, `anglediff`, `keygroups` |

---

## Workflow

The three notebooks in [`notebooks/`](notebooks) run the full reduction in order. At the top of each notebook, set `DATA_DIR` (or `data_dir` / `dest_dir`) to your own folders.

### 1. Sort raw data: [`sphere_sort_hopsets.ipynb`](notebooks/sphere_sort_hopsets.ipynb)

Reads the headers of every raw `SPH*.fits` file in `data_dir`, groups them by observation container, and writes one folder per dataset to `dest_dir`:

```
SORT/dayYYYY-MM-DD-cidNNNNNNN-IRDIS/
├── scilist.txt     sciinfo.txt
├── flatlist.txt    flatinfo.txt
├── skylist.txt     skyinfo.txt
├── darklist.txt    darkinfo.txt
├── centerlist.txt  centerinfo.txt
└── fluxlist.txt    fluxinfo.txt
```

```python
from starhoppingpy import sh_sort as shs

entries  = [shs.parse_dfits_line(r) for r in shs.get_file_list("/path/to/raw")]
datasets = shs.group_datasets(entries)
for ds_id, ds_entries in datasets.items():
    ds_dir = shs.process_science_files(ds_id, ds_entries, "/path/to/raw", "/path/to/SORT", "msglog.txt")
    shs.process_calibration_flat(entries, ds_entries[0], "/path/to/raw", ds_dir, "msglog.txt")
    # ... sky, dark, center, flux likewise
```

### 2. Calibrate and do the basic reduction: [`irdis_basic_reduc_starhop.ipynb`](notebooks/irdis_basic_reduc_starhop.ipynb)

Run inside one sorted dataset folder:

```python
import os
from starhoppingpy.sh_funcs import (
    irdis_makeflat, irdis_make_sky, irdis_badpix_from_obj2, sph_cen_irdis,
    irdis_basicreduc, irdis_make_asdi_cube, sph_remove_bad_images,
)

os.chdir("/path/to/SORT/dayYYYY-MM-DD-cidNNNNNNN-IRDIS")

irdis_makeflat("flatlist.txt")            # -> flat_irdis.fits, badpix_irdis.fits
irdis_make_sky("skylist.txt")             # -> sky_irdis.fits
sph_cen_irdis("centerlist.txt")           # -> centers.save.npz (star center per channel)
irdis_basicreduc("scilist.txt")           # -> basic-reduced frames + basicredlist{1,2}.txt

# 1 = science star, 2 = reference star
irdis_make_asdi_cube("basicredlist1.txt", outfile="imcbasic1.save", szn=501)
irdis_make_asdi_cube("basicredlist2.txt", outfile="imcbasic2.save", szn=501)

# reject poor frames using residual RMS in two annuli
for f in ("imcbasic1.save.npz", "imcbasic2.save.npz"):
    sph_remove_bad_images(f, inrad=8,  outrad=40)
    sph_remove_bad_images(f, inrad=40, outrad=200)
```

Each `imcbasic*.save.npz` file holds the image cube (`imc`), the parallactic angles (`pas`), and per-frame metadata such as `detside`.

### 3. Star-hopping RDI with LOCI: [`double_loci_restrict3_npz.ipynb`](notebooks/double_loci_restrict3_npz.ipynb)

1. **Rescale the reference PSF** to match the science PSF:
   `sph_rescale_ref_psf("imcbasic1.save.npz", "imcbasic2.save.npz", "imcbasic2b.save.npz")`
2. **Define the optimization regions** with `basic.get_mask` and remove background slopes with `functions.remove_bgnd_slope`.
3. *(Point sources only)* **High-pass filter** each frame with `zfuncs.getfreq`. Skip this step for disks and other extended sources.
4. **First-pass LOCI.** For each science frame, `functions.loci_operation` builds the best linear combination of reference frames. The difference is de-rotated to north-up and median-combined.
5. **Build signal and background masks** from the first-pass image and an annular RMS map, so the LOCI fit is driven only by background pixels and does not self-subtract real signal.
6. **Second-pass LOCI.** Reference frames are ranked for each science frame (`functions.good_refs_mask`), the subtraction is redone with the masks, and each frame gets an SNR score.
7. **Stack** the best frames (SNR > 0.65 × max) into the final `<name>_red1.fits`.

---

## Quick example

```python
import numpy as np
from starhoppingpy import basic, functions, zfuncs as zf
from scipy import ndimage

s = np.load("imcbasic1.save.npz",  allow_pickle=True)
r = np.load("imcbasic2b.save.npz", allow_pickle=True)
cubeS, cubeR, angs = s["imc"], r["imc"], s["pas"]

wsb = basic.get_mask(cubeS[0], 6, 120)          # optimisation region (pixels)
cubeS = functions.remove_bgnd_slope(cubeS)
cubeR = functions.remove_bgnd_slope(cubeR)

diffs = np.empty_like(cubeS)
for i, img in enumerate(cubeS):
    ref = functions.loci_operation(img, cubeR, wsb)
    diffs[i] = ndimage.rotate(img - ref, angs[i], reshape=False)

final = np.median(diffs, axis=0)
zf.writefits("target_rdi.fits", final)
zf.zdisp(final, limfrac=0.1)
```

---

## Notes and tips

- **Data are not tracked.** `.gitignore` excludes `*.fits`, `*.fits.Z`, `*.fits.gz`, `*.npz` and `*.save*`. Keep observations and products outside the repository, or rely on these rules.
- **Compressed archive files.** `zfuncs.readfits` reads `.fits.Z` files directly when `unlzw3` is installed. To sort compressed raw data, change the glob pattern in `sh_sort.get_file_list` to match `SPH*.fits.Z`.
- **Working directory.** Most `sh_funcs` routines read and write fixed filenames (`flat_irdis.fits`, `sky_irdis.fits`, `centers.save.npz`, …) in the current directory, so `os.chdir` into the dataset folder first.
- **NDIT mismatches.** The basic-reduction notebook has a check for files whose header `NDIT` does not match the cube shape. Use it if `irdis_make_asdi_cube` fails on a file.

---

## Citation

If you use this code, please cite the star-hopping method paper:

> Wahhaj, Z., Milli, J., Romero, C., et al. 2021, *A&A*, 648, A26 — *"A search for a 5th planet around HR 8799 using the star-hopping RDI technique at VLT/SPHERE"*

```bibtex
@article{Wahhaj2021,
  author  = {{Wahhaj}, Z. and {Milli}, J. and {Romero}, C. and others},
  title   = {A search for a 5th planet around HR 8799 using the star-hopping RDI technique at VLT/SPHERE},
  journal = {Astronomy \& Astrophysics},
  year    = {2021},
  volume  = {648},
  pages   = {A26},
  adsurl  = {https://ui.adsabs.harvard.edu/abs/2021A%26A...648A..26W}
}
```

## Credits

The star-hopping technique and the original IDL reduction scripts are by Z. Wahhaj and collaborators. This Python port is maintained by Swastik Chowbay.

## License

Released under the [MIT License](LICENSE).
