# starhoppingpy

**A Python pipeline for star-hopping reference differential imaging (RDI) with VLT/SPHERE-IRDIS.**

`starhoppingpy` takes raw ESO SPHERE-IRDIS frames and turns them into a final PSF-subtracted image of the circumstellar environment. Point it at a folder of raw `SPHER.*.fits` files and it will:

1. sort them into science and calibration sets,
2. build the flat, sky, dark and bad-pixel calibrations,
3. find the star center behind the coronagraph,
4. calibrate every science frame,
5. split the frames into **science-star** and **reference-star** cubes, with parallactic angles,
6. reject bad frames,
7. subtract the stellar PSF using the reference star with LOCI, and
8. de-rotate and stack the result into a FITS image.

It is a Python port of the original IDL star-hopping reduction scripts, and keeps their file names and processing logic.

---

## Table of contents

- [What is star-hopping?](#what-is-star-hopping)
- [Pipeline overview](#pipeline-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Step-by-step guide](#step-by-step-guide)
  - [Step 1: Sort the raw data](#step-1-sort-the-raw-data)
  - [Step 2: Build the calibrations](#step-2-build-the-calibrations)
  - [Step 3: Find the star center](#step-3-find-the-star-center)
  - [Step 4: Basic reduction](#step-4-basic-reduction)
  - [Step 5: Build the image cubes](#step-5-build-the-image-cubes)
  - [Step 6: Reject bad frames](#step-6-reject-bad-frames)
  - [Step 7: Star-hopping RDI with LOCI](#step-7-star-hopping-rdi-with-loci)
- [Output file reference](#output-file-reference)
- [Module and function reference](#module-and-function-reference)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Citation](#citation)
- [Credits](#credits)
- [License](#license)

---

## What is star-hopping?

High-contrast imaging must remove the bright stellar PSF to reveal faint planets and disks. The usual approach, **angular differential imaging (ADI)**, relies on field rotation during the night. It works poorly close to the star and for extended or face-on disks, because the rotation also smears out and subtracts the real signal.

**Star-hopping** ([Wahhaj et al. 2021, A&A 648, A26](https://ui.adsabs.harvard.edu/abs/2021A%26A...648A..26W)) is a form of **reference differential imaging (RDI)** designed for SPHERE. During a single observation, the telescope switches back and forth between the **science target** and a nearby **reference star** of similar brightness and color, with only about a minute of overhead per hop. Both stars are observed under almost the same conditions and AO correction, so the reference PSF is a close match to the science PSF and can be subtracted directly, with no need for field rotation.

`starhoppingpy` reduces exactly this kind of data set: one observing block that contains interleaved science-star and reference-star frames.

---

## Pipeline overview

```
 raw SPHER.*.fits
        │
        ▼
 ┌────────────────────┐   sh_sort.*                    → SORT/day<date>-cid<ID>-IRDIS/
 │ 1. Sort            │                                   scilist.txt, flatlist.txt, skylist.txt, …
 └────────────────────┘
        │
        ▼
 ┌────────────────────┐   irdis_makeflat               → flat_irdis.fits, badpix_irdis.fits
 │ 2. Calibrations    │   irdis_make_sky               → sky_irdis.fits
 └────────────────────┘
        │
        ▼
 ┌────────────────────┐   sph_cen_irdis                → centers.save.npz, dark_irdis.fits
 │ 3. Star center     │
 └────────────────────┘
        │
        ▼
 ┌────────────────────┐   irdis_basicreduc             → basicSPHER.*.fits
 │ 4. Basic reduction │                                   basicredlist1.txt (science star)
 └────────────────────┘                                   basicredlist2.txt (reference star)
        │
        ▼
 ┌────────────────────┐   irdis_make_asdi_cube         → imcbasic1.save.npz (science cube)
 │ 5. Image cubes     │                                   imcbasic2.save.npz (reference cube)
 └────────────────────┘
        │
        ▼
 ┌────────────────────┐   sph_remove_bad_images        → same .npz files, bad frames removed
 │ 6. Frame rejection │
 └────────────────────┘
        │
        ▼
 ┌────────────────────┐   sph_rescale_ref_psf          → imcbasic2b.save.npz
 │ 7. RDI with LOCI   │   functions.loci_operation, …  → <name>_red1.fits  (final image)
 └────────────────────┘
```

Steps 1–6 are in the package. Step 7 is run from the notebook [`double_loci_restrict3_npz.ipynb`](notebooks/double_loci_restrict3_npz.ipynb), which calls package functions.

---

## Requirements

### Software

| Requirement | Version / notes |
| --- | --- |
| Python | **3.9 or newer** |
| `numpy`, `scipy`, `astropy`, `matplotlib` | Installed automatically |
| ESO **`dfits`** and **`fitsort`** | Command-line tools needed **only for Step 1 (sorting)**. They are distributed with the ESO pipeline tools (`eclipse`/`qfits`) and must be on your `PATH`. |
| `unlzw3` *(optional)* | Needed to read compressed archive files (`*.fits.Z`) |
| `scikit-image` *(optional)* | Image resizing in `basic.py` |
| `fpdf` *(optional)* | `basic.pdf_creator` |
| `pysynphot` *(optional, not in extras)* | `zfuncs.zp` photometry. Also needs the `PYSYN_CDBS` reference files. |
| `jupyter`, `ipywidgets` *(optional)* | Running the notebooks |

### Data

- **Instrument:** SPHERE **IRDIS** (classical imaging / DBI). The sorting step also recognises IFS calibrations, but every reduction function after sorting is IRDIS-only.
- **Observation type:** a star-hopping sequence. Science and reference frames must be in the **same observation container** (`OBS.CONTAINER.ID`).
- **File names:** the ESO archive convention `SPHER.YYYY-MM-DDThh:mm:ss.sss.fits`. The sorting step reads the date and time of each file from its name.
- **Calibrations needed in the same raw folder:**
  - `FLAT`
  - `SKY` (or `DARK,BACKGROUND`)
  - `OBJECT,CENTER` (satellite-spot frames)
  - `OBJECT,FLUX` *(optional)*

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/SwastikC/starhoppingpy.git
cd starhoppingpy
```

**2. (Recommended) create an isolated environment**

```bash
python -m venv .venv
source .venv/bin/activate
```

**3. Install the package.** `-e` installs in editable mode, so your edits to the source take effect without reinstalling.

```bash
pip install -e .
```

**4. Install the optional extras you need**

```bash
pip install -e ".[extras]"      # unlzw3, scikit-image, fpdf
pip install -e ".[notebooks]"   # jupyter, ipywidgets
```

**5. Check the installation**

```bash
python -c "import starhoppingpy; print(starhoppingpy.__version__)"
```

This should print `0.1.0`. If you are going to sort raw data, also check that the ESO tools are found:

```bash
which dfits fitsort
```

---

## Quick start

The shortest complete run is below. Every call is explained in the [step-by-step guide](#step-by-step-guide).

```python
import os
from starhoppingpy import sh_sort as shs
from starhoppingpy.sh_funcs import (
    irdis_makeflat, irdis_make_sky, sph_cen_irdis, irdis_basicreduc,
    irdis_make_asdi_cube, sph_remove_bad_images, sph_rescale_ref_psf,
)

RAW  = "/data/sphere/raw"     # folder with raw SPHER.*.fits
SORT = "/data/sphere/SORT"    # where sorted datasets will be created

# --- Step 1: sort ---------------------------------------------------------
entries  = [shs.parse_dfits_line(row) for row in shs.get_file_list(RAW)]
datasets = shs.group_datasets(entries)
for ds_id, ds_entries in datasets.items():
    ds_dir = shs.process_science_files(ds_id, ds_entries, RAW, SORT, "msglog.txt")
    sci = ds_entries[0]
    for step in (shs.process_calibration_flat, shs.process_calibration_sky,
                 shs.process_calibration_dark, shs.process_calibration_center,
                 shs.process_calibration_flux):
        step(entries, sci, RAW, ds_dir, "msglog.txt")

# --- Steps 2-6: run inside ONE dataset folder -----------------------------
os.chdir(os.path.join(SORT, "day2021-05-03-cid1234567-IRDIS"))   # <- your folder name

irdis_makeflat("flatlist.txt")
irdis_make_sky("skylist.txt")
sph_cen_irdis("centerlist.txt")
irdis_basicreduc("scilist.txt")
irdis_make_asdi_cube("basicredlist1.txt", outfile="imcbasic1.save", szn=501)
irdis_make_asdi_cube("basicredlist2.txt", outfile="imcbasic2.save", szn=501)
for f in ("imcbasic1.save.npz", "imcbasic2.save.npz"):
    sph_remove_bad_images(f, inrad=8,  outrad=40)
    sph_remove_bad_images(f, inrad=40, outrad=200)

# --- Step 7: start of the RDI step ----------------------------------------
sph_rescale_ref_psf("imcbasic1.save.npz", "imcbasic2.save.npz", "imcbasic2b.save.npz")
# ...then open notebooks/double_loci_restrict3_npz.ipynb for the LOCI subtraction
```

---

## Step-by-step guide

> **Important:** from Step 2 onward, every function reads and writes **fixed file names in the current working directory**. Always `os.chdir()` into the sorted dataset folder first, and run the steps **in order**, because each step needs files written by the earlier ones.

### Step 1: Sort the raw data

**Notebook:** [`sphere_sort_hopsets.ipynb`](notebooks/sphere_sort_hopsets.ipynb)
**Module:** `starhoppingpy.sh_sort`

**What it does**

1. `get_file_list(data_dir)` runs `dfits | fitsort` on every `SPH*.NNN.fits` file in `data_dir` and returns one tab-separated header line per file. It reads these keywords:

   | Keyword | Stored as |
   | --- | --- |
   | `OBS.NAME` | `object` |
   | `INS4.MODE` | `ins4_mode` |
   | `SEQ.ARM` | `arm` (`IRDIS` / `IFS`) |
   | `DPR.TYPE` | `dprtype` (e.g. `OBJECT`, `OBJECT,CENTER`, `SKY`, `FLAT,LAMP`) |
   | `INS.COMB.ICOR` | `ins_comb_icor` (coronagraph) |
   | `INS1.FILT.NAME` | `ins1_filt_name` |
   | `INS.COMB.IFLT` | `filt` (filter combination) |
   | `INS2.OPTI2.NAME` | `prism` |
   | `DET.SEQ1.DIT`, `DET.NDIT` | `dit`, `ndit` |
   | `OBS.CONTAINER.ID` | `contid` |
   | `ORIGFILE`, `INS2.COMB.IFS` | `origfile`, `ifslamp` |

2. `parse_dfits_line(line)` turns each line into a Python `dict` with the fields above.
3. `group_datasets(entries)` groups the non-calibration files (those whose `OBS.NAME` does not contain `Calibration`) into datasets keyed by **date + container ID + arm**: `day<YYYY-MM-DD>-cid<containerID>-<ARM>`.
4. `process_science_files(...)` creates the dataset folder under `dest_dir` and writes `scilist.txt` / `sciinfo.txt`. It also **symlinks** the raw files into the folder, so the raw data are not copied.
5. Each `process_calibration_*` function picks the matching calibrations from **all** raw files, writes `<type>list.txt` (file names) and `<type>info.txt` (details, including time offset), and symlinks them into the folder:

   | Function | Picks files with `DPR.TYPE` containing | Must also match the science file in | Time selection |
   | --- | --- | --- | --- |
   | `process_calibration_flat` | `FLAT` | `prism`, `arm`, container ID* | the single closest flat (warns if > 2 days away) |
   | `process_calibration_sky` | `SKY` | `filt` (IRDIS) or `prism` (IFS), `arm`, container ID* | all skies within 16 h of the closest one |
   | `process_calibration_dark` | `DARK,BACK` | `filt`, `arm`, **DIT**, container ID* | all darks within 16 h of the closest one |
   | `process_calibration_center` | `CENTER` | `filt`, `arm`, **DIT**, container ID* | all within 16 h of the closest one |
   | `process_calibration_flux` | `FLUX` | `arm`, container ID* | all within 4 h of the closest one (warns if > 2 h) |
   | `process_calibration_ifscal`, `process_calibration_ifscal2` | IFS lamp calibrations | IFS only (skipped for IRDIS) | see docstrings |

   \* A calibration with an **empty** container ID is also accepted.

**How to run it**

```python
from starhoppingpy import sh_sort as shs

data_dir = "/path/to/raw"         # raw SPHER.*.fits (uncompressed)
dest_dir = "/path/to/SORT"        # will be created/filled
msgfile  = "msglog.txt"

entries  = [shs.parse_dfits_line(row) for row in shs.get_file_list(data_dir)]
datasets = shs.group_datasets(entries)

for ds_id, ds_entries in datasets.items():
    print(ds_id, len(ds_entries), "files")
    ds_dir = shs.process_science_files(ds_id, ds_entries, data_dir, dest_dir, msgfile)
    sci = ds_entries[0]
    shs.process_calibration_flat  (entries, sci, data_dir, ds_dir, msgfile)
    shs.process_calibration_sky   (entries, sci, data_dir, ds_dir, msgfile)
    shs.process_calibration_dark  (entries, sci, data_dir, ds_dir, msgfile)
    shs.process_calibration_center(entries, sci, data_dir, ds_dir, msgfile)
    shs.process_calibration_flux  (entries, sci, data_dir, ds_dir, msgfile)
```

**Result:** one folder per dataset.

```
SORT/
└── day2021-05-03-cid1234567-IRDIS/
    ├── scilist.txt      sciinfo.txt      # all target frames (science + reference star)
    ├── flatlist.txt     flatinfo.txt
    ├── skylist.txt      skyinfo.txt
    ├── darklist.txt     darkinfo.txt
    ├── centerlist.txt   centerinfo.txt
    ├── fluxlist.txt     fluxinfo.txt
    └── SPHER.*.fits  -> symlinks to the raw files
```

**Check before you continue:** open each `*list.txt` and make sure it is not empty. If one is empty, the function prints `No calibration ... candidates found`. Add the missing calibration to the raw folder, or write that list file by hand.

> `scilist.txt` contains **every** frame of the observation, including `OBJECT,CENTER`, `OBJECT,FLUX` and `SKY` frames, and frames of both the science and the reference star. Later steps pick out the `OBJECT` frames and split the two stars automatically.

---

### Step 2: Build the calibrations

**Notebook:** [`irdis_basic_reduc_starhop.ipynb`](notebooks/irdis_basic_reduc_starhop.ipynb), cells 1–3
**Module:** `starhoppingpy.sh_funcs`

```python
import os
from starhoppingpy.sh_funcs import irdis_makeflat, irdis_make_sky, irdis_badpix_from_obj2

os.chdir("/path/to/SORT/day2021-05-03-cid1234567-IRDIS")

irdis_makeflat("flatlist.txt")
irdis_make_sky("skylist.txt")
```

| Function | Input | What it does | Writes |
| --- | --- | --- | --- |
| `irdis_makeflat(filetxt)` | `flatlist.txt` | Median-collapses the flat cube and normalizes it by its median. If two flats with different DITs are listed, it flags as bad (non-linear) every pixel whose flat ratio differs from the DIT ratio by more than 10%. It sets the detector edges and the central gap to 1, interpolates over bad pixels, clips the flat to 0.8–1.2, and flags pixels outside that range as bad. Uses at most the first two files. | `flat_irdis.fits`, `badpix_irdis.fits` (plus intermediate `flat_irdis0*.fits`, `badpix_irdis0*.fits`) |
| `irdis_make_sky(filelist, show=False)` | `skylist.txt` | Pixel-by-pixel median of all sky frames. Set `show=True` to display each frame. | `sky_irdis.fits` |
| `irdis_badpix_from_obj2(filetxt, iter=3)` *(optional)* | `scilist.txt` | Builds a bad-pixel map from the first science cube instead of the flat. Use this if the flat-based map is poor. **This overwrites `badpix_irdis.fits`.** | `badpix_irdis.fits`, `temp.fits`, `sci_bp_clean_example.fits` |

**Check:** display `flat_irdis.fits` (values around 1) and `badpix_irdis.fits` (1 = bad pixel):

```python
from starhoppingpy import zfuncs as zf
zf.zdisp(zf.readfits("flat_irdis.fits"))
```

---

### Step 3: Find the star center

**Notebook:** `irdis_basic_reduc_starhop.ipynb`, cell 4

The star sits behind the coronagraph, so its position is measured from the four **satellite spots** in the `OBJECT,CENTER` frames.

```python
from starhoppingpy.sh_funcs import sph_cen_irdis
sph_cen_irdis("centerlist.txt", lmaskoutw0=30, rmaskoutw0=30)
```

**What it does, for each center frame:**

1. It median-collapses the frame. It subtracts the first file in `skylist.txt` (or `darklist.txt` if there is no sky list) and saves that file as **`dark_irdis.fits`**, which Step 4 needs.
2. It fixes bad pixels using `badpix_irdis.fits`.
3. It splits the detector into its **left and right halves**. In each half, it finds the 4 brightest spots near the expected center (taken from the header) and centroids them with `basic.bscentrd`.
4. It averages the four spot positions to get the star center for each half.
5. If a center is more than 20 px from the expected position, it prints `Centering algo failed. Check Manually.` and skips that frame.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `filetxt` | — | **Ignored.** The function always reads `centerlist.txt`. |
| `darkfiletxt` | auto | Dark/sky list. Defaults to `skylist.txt`, then `darklist.txt`. |
| `badpix_file` | `badpix_irdis.fits` | Bad-pixel map to use |
| `lmaskoutw0`, `rmaskoutw0` | 30 | Inner edge (px) of the spot-search annulus. Everything closer to the star is masked (left / right half). |
| `lmaskinw1`, `rmaskinw1` | 100 | Outer edge (px) of the spot-search annulus. Everything farther out is masked. |

**Writes:** `centers.save.npz`, which contains the median centers `xc1, yc1` (left) and `xc2, yc2` (right) and the per-frame values `xcs1, ycs1, xcs2, ycs2`. It also writes `dark_irdis.fits`.

**Check:** `np.load("centers.save.npz")["xcs1"]` should show consistent values. Entries of `-1` are frames where centering failed.

---

### Step 4: Basic reduction

**Notebook:** `irdis_basic_reduc_starhop.ipynb`, cell 7

```python
from starhoppingpy.sh_funcs import irdis_basicreduc
irdis_basicreduc("scilist.txt")       # optional: maskfile=..., tag="_v2"
```

**Needs** these files in the current folder: `flat_irdis.fits`, `sky_irdis.fits`, `dark_irdis.fits`, `badpix_irdis.fits`.

**What it does:**

1. Reads each file in `scilist.txt` and **keeps only `DPR.TYPE = OBJECT`** frames.
2. For every DIT in the cube, computes `(frame − sky) / flat`.
3. Builds a per-file bad-pixel map: outliers relative to a median-filtered image, combined with `badpix_irdis.fits`. It then corrects the bad pixels with `basic.fixpix`.
4. Sets to zero the detector edges and the gap between the two IRDIS halves (columns `< 50`, `940–1080` and `≥ 1960`).
5. Saves each result as `basic<original name>.fits`. A `.Z` extension is dropped.
6. **Splits science and reference stars automatically,** using the telescope `RA`/`DEC` in each header. Frames within 0.001° (3.6″) of the first frame's position form one group and the rest form the other. **The group with more frames is taken to be the science star.**

**Writes:**

| File | Contents |
| --- | --- |
| `basicSPHER.*.fits` | Calibrated frames |
| `basicredlist.txt` | All calibrated frames |
| `basicredlist1.txt` | **Science-star** frames (the larger group) |
| `basicredlist2.txt` | **Reference-star** frames (the smaller group) |

> ⚠️ If your reference star has **more** frames than the science star, the two lists are swapped. Check them against `sciinfo.txt` (column `OBJECT`) and rename them if needed.

Optional `maskfile`: a `.npy` file holding a dict with key `wmask` (a boolean image). The pixels it marks are replaced by noise before processing.

---

### Step 5: Build the image cubes

**Notebook:** `irdis_basic_reduc_starhop.ipynb`, cells 9–11

```python
from starhoppingpy.sh_funcs import irdis_make_asdi_cube
irdis_make_asdi_cube("basicredlist1.txt", outfile="imcbasic1.save", szn=501)   # science star
irdis_make_asdi_cube("basicredlist2.txt", outfile="imcbasic2.save", szn=501)   # reference star
```

**What it does:**

1. Computes flux-normalization factors (`irdis_normfluxfac`) and writes them to `normscifac.save`.
2. Loads the median star centers from `centers.save.npz`.
3. Splits every DIT into its **left and right** IRDIS images. It shifts each one so the star is in the exact center, correcting for the dither offsets `INS1 DITH POSX/POSY`, and crops it to `szn × szn` pixels.
4. Gives each DIT a parallactic angle, interpolated linearly between `TEL PARANG START` and `TEL PARANG END`.
5. Gives each image a wavelength based on the filter:

   | Filter (`INS1 FILT ID` + `INS1 OPTI2 ID`) | λ left / right (µm) |
   | --- | --- |
   | `BB_H` + `CLEAR` | 1.625 / 1.625 |
   | `BB_H` + `DBF_H23` | 1.593 / 1.667 |
   | `BB_H` + `DBF_H34` | 1.667 / 1.733 |
   | `BB_Y` + `DBF_Y23` | 1.022 / 1.076 |
   | `BB_J` + `DBF_J23` | 1.190 / 1.273 |
   | `BB_Ks` + `DBF_K12` | 2.110 / 2.251 |
   | other broadband Y / J / H / Ks | 1.043 / 1.245 / 1.625 / 2.182 |

**Writes:** `<outfile>.npz`. NumPy adds the `.npz` extension, so `outfile="imcbasic1.save"` gives `imcbasic1.save.npz`. Its contents:

| Key | Shape | Meaning |
| --- | --- | --- |
| `imc` | `(N, szn, szn)` | Centered images, 2 per DIT (left and right) |
| `detside` | `(N,)` | `0` = left half, `1` = right half |
| `pas` | `(N,)` | Parallactic angle (deg, sign-flipped for de-rotation) |
| `lambdas` | `(N,)` | Wavelength of each image (µm) |
| `dits` | `(N,)` | Exposure time per frame (s) |
| `tstamps` | `(N,)` | Time stamps |
| `stokes`, `polfilts` | `(N,)` | Polarimetry info (placeholders for imaging) |
| `normfac` | | Flux-normalization factors |
| `lambdas0`, `nlambda` | | Unique wavelengths and how many there are |

**Load it:**

```python
import numpy as np
d = np.load("imcbasic1.save.npz", allow_pickle=True)
cube, angles = d["imc"], d["pas"]
print(cube.shape, "total rotation:", angles.max() - angles.min(), "deg")
```

> If this step fails on a file, the `NDIT` header value may not match the real cube size. Cell 8 of the notebook lists the affected files so you can remove them from `basicredlist1.txt`.

---

### Step 6: Reject bad frames

**Notebook:** `irdis_basic_reduc_starhop.ipynb`, cells 12–13

```python
import shutil
from starhoppingpy.sh_funcs import sph_remove_bad_images

# 1) keep a backup, because the files are overwritten in place
shutil.copy("imcbasic1.save.npz", "imcbasic1.save.org.npz")
shutil.copy("imcbasic2.save.npz", "imcbasic2.save.org.npz")

# 2) reject frames using an inner and an outer annulus
for f in ("imcbasic1.save.npz", "imcbasic2.save.npz"):
    sph_remove_bad_images(f, inrad=8,  outrad=40)    # close to the coronagraph
    sph_remove_bad_images(f, inrad=40, outrad=200)   # AO-corrected halo
```

**What it does:** for each wavelength separately, it subtracts the median image of the cube from every frame and measures the RMS inside the annulus `inrad ≤ r ≤ outrad` (pixels). A frame is **rejected** if its RMS is above `median + 2σ` of all the frame RMS values. The rejected frames are removed from every array in the `.npz` file, and the file is **saved over the original**.

Use `fullres=True` to get the good and bad indices and the RMS values back.

---

### Step 7: Star-hopping RDI with LOCI

**Notebook:** [`double_loci_restrict3_npz.ipynb`](notebooks/double_loci_restrict3_npz.ipynb)

This step is interactive: it shows images after each stage so you can tune the masks. Before running, set at the top of the notebook:

- `name`: the target name, used for the output file name (e.g. `name = 'hd141569'`).
- `DATA_DIR`: the sorted dataset folder.

**7a. Rescale the reference PSF to the science PSF**

```python
from starhoppingpy.sh_funcs import sph_rescale_ref_psf
sph_rescale_ref_psf("imcbasic1.save.npz", "imcbasic2.save.npz", "imcbasic2b.save.npz")
```

The reference star is never identical to the science star. This step fits a power-law radial profile between the coronagraph edge (r = 8 px) and the AO ring (r = 66 px), using the median background at 120–140 px. It then multiplies every reference frame by the ratio of the two profiles. It plots the science, reference and difference radial profiles. **Writes** `imcbasic2b.save.npz` (same keys as the input).

**7b. Load the cubes and define regions**

All regions are annuli centered on the star, given as `basic.get_mask(image, r_in, r_out)` in **pixels**. One IRDIS pixel is about 12.25 mas.

| Variable | Radii (px) | Used for |
| --- | --- | --- |
| `w0` | 0–4 | Region behind the coronagraph |
| `w1` | 70–110 | Outer region for PSF scaling |
| `w2` | 90–120 | Alternative outer region |
| `wb` | 120–240 | Background |
| `wsb` | 6–120 | Signal + some background: the main LOCI optimization region |

`functions.remove_bgnd_slope` then removes a planar (1st-order) background from every frame.

**7c. (Point sources only) Spatial filter.** `zf.getfreq(img, 4)` is a Gaussian band-pass filter for a PSF FWHM of 4 px. It removes pixel-scale noise (below FWHM/2 = 2 px) and smooth structure (above 2×FWHM = 8 px), so point sources stand out. **Skip this cell for disks or other extended sources.**

**7d. First LOCI pass.** For each science frame, `functions.loci_operation(frame, ref_cube, wsb)` finds the least-squares linear combination of reference frames that best matches the frame inside `wsb`. That combination is subtracted, and the residual is de-rotated by its parallactic angle. The median of all residuals is the first-pass image.

**7e. Signal and background masks.** The first-pass image is compared with an annular RMS map of the reference residuals:

- `bmask` (background): pixels below `lev × RMS` (`lev = 1`) between 12 and 110 px.
- `smask` (signal): pixels above `lev2 × RMS` (`lev2 = 40`).

The background mask is rotated into each frame's orientation and averaged into `nmask`. This means the next fit is driven **only by background pixels**, which keeps LOCI from subtracting the real astrophysical signal (self-subtraction). For faint sources, lower `lev2`.

**7f. Remove the mean PSF per detector side.** For each detector half (`detside`), the median science PSF is fitted with the best 16 reference frames (`functions.good_refs2` + `functions.loci2`). That model is then removed from every science and reference frame with `functions.zscalediff2`.

**7g. Second LOCI pass.** `functions.good_refs_mask` ranks the reference frames for each science frame using the background mask. The best `nrefs = min(N_ref − 1, 16)` frames are used for the LOCI fit inside the rotated background mask. Each residual gets a signal-to-noise score, `sum(signal mask) / std(background mask)`.

**7h. Stack.** Frames with an SNR above `0.65 × max(SNR)` are median-combined after de-rotation. The central 8 px are set to zero, and the result is written to **`<name>_red1.fits`**.

**7i. (Optional) Iterate.** The last cell builds new masks from the final image (`lev2 = 5`, radius 12–120 px), so you can run 7g–7h again.

**Parameters you will most likely tune:**

| Parameter | Where | Default | Effect |
| --- | --- | --- | --- |
| `wsb` radii | 7b | 6–120 px | Region used to fit the PSF |
| `getfreq` FWHM | 7c | 4 px | PSF size the band-pass filter is tuned to |
| `lev`, `lev2` | 7e | 1, 40 | Thresholds for background and signal masks |
| `numref` / `nrefs` | 7f, 7g | 16 | Number of reference frames per fit |
| SNR cut | 7h | 0.65 × max | Fraction of frames kept in the final stack |

---

## Output file reference

Every file the pipeline writes inside a dataset folder:

| File | Written by | Step |
| --- | --- | --- |
| `scilist.txt`, `sciinfo.txt` | `process_science_files` | 1 |
| `flatlist.txt`, `skylist.txt`, `darklist.txt`, `centerlist.txt`, `fluxlist.txt` (+ `*info.txt`) | `process_calibration_*` | 1 |
| `flat_irdis.fits`, `badpix_irdis.fits` | `irdis_makeflat` | 2 |
| `sky_irdis.fits` | `irdis_make_sky` | 2 |
| `centers.save.npz`, `dark_irdis.fits` | `sph_cen_irdis` | 3 |
| `basicSPHER.*.fits`, `basicredlist.txt`, `basicredlist1.txt`, `basicredlist2.txt` | `irdis_basicreduc` | 4 |
| `normscifac.save.npz` | `irdis_normfluxfac` (called by `irdis_make_asdi_cube`) | 5 |
| `imcbasic1.save.npz`, `imcbasic2.save.npz` | `irdis_make_asdi_cube` | 5 |
| `imcbasic2b.save.npz` | `sph_rescale_ref_psf` | 7 |
| `<name>_red1.fits` | LOCI notebook | 7 |
| `flux.save.npz` | `irdis_make_flux_cube` (optional) | — |

---

## Module and function reference

```
src/starhoppingpy/
├── sh_sort.py      # Step 1: sorting raw data
├── sh_funcs.py     # Steps 2–7a: calibrations, centering, reduction, cubes
├── basic.py        # low-level image tools
├── functions.py    # LOCI / RDI subtraction
└── zfuncs.py       # FITS I/O, display and general helpers
```

<details>
<summary><b><code>sh_sort</code>: sorting</b></summary>

| Function | Purpose |
| --- | --- |
| `get_file_list(data_dir)` | Run `dfits \| fitsort` on `SPH*.NNN.fits` files and return the header lines |
| `parse_dfits_line(line)` | Convert one header line into a dict |
| `group_datasets(entries)` | Group science files by date, container ID and arm |
| `process_science_files(id, entries, data_dir, dest_dir, msgfile)` | Create the dataset folder, `scilist.txt` and symlinks |
| `process_calibration_flat/sky/dark/center/flux(...)` | Select and link matching calibrations |
| `process_calibration_ifscal/ifscal2(...)` | IFS calibrations (IFS data only) |
| `print_science_summary(dest_dir)` | Print a summary of sorted datasets |

`sh_funcs` contains the same sorting functions, but its `get_file_list` looks for **compressed** `SPH*.NNN.fits.Z` files.

</details>

<details>
<summary><b><code>sh_funcs</code>: calibrations and reduction</b></summary>

| Function | Purpose |
| --- | --- |
| `irdis_makeflat(filetxt)` | Flat field and bad-pixel map |
| `irdis_make_sky(filelist, show=False)` | Median sky |
| `irdis_badpix_from_obj2(filetxt, iter=None)` | Bad-pixel map from science data |
| `qzap(img, ...)` | Cosmic-ray / hot-pixel removal |
| `sph_cen_irdis(filetxt, ...)` | Star center from satellite spots |
| `irdis_make_flux_cube(fluxlist, szn=None)` | Cube of unsaturated flux (PSF) frames |
| `irdis_normfluxfac(filelist, ...)` | Flux-normalization factors (DIT, ND filter) |
| `irdis_basicreduc(scilist, maskfile=None, tag='')` | Calibrate frames; split science and reference |
| `irdis_make_asdi_cube(basicredlist, szn=None, outfile=None, ...)` | Centered image cube with angles and wavelengths |
| `get_good_images(imc, inrad, outrad, cut_factor=None)` | Find frames with outlier RMS |
| `sph_remove_bad_images(insavefile, inrad, outrad)` | Remove those frames from an `.npz` cube |
| `sph_rescale_ref_psf(sci, ref, out)` | Match the reference PSF profile to the science PSF |

</details>

<details>
<summary><b><code>functions</code>: LOCI / RDI</b></summary>

| Function | Purpose |
| --- | --- |
| `remove_bgnd_slope(cube)` | Subtract a fitted plane from each frame |
| `remove_median_in_region(cube, ww)` | Subtract the median inside a region |
| `loci_operation(image, ref_cube, wm)` | Best linear combination of references inside mask `wm` |
| `loci2(image, ref_cube, w1, w2)` | LOCI with two regions balanced by `rgn_fac` |
| `rgn_fac(image, w1, w2)` | Weighting factor between two regions |
| `best_refs`, `good_refs`, `good_refs2`, `good_refs_mask` | Rank reference frames for each science frame |
| `zscalediff`, `zscalediff2` | Scale and subtract a reference within regions |
| `sublcomb1d`, `sublcomb1dm` | Core least-squares linear-combination solvers |

</details>

<details>
<summary><b><code>basic</code>: image tools</b></summary>

| Function | Purpose |
| --- | --- |
| `get_mask(img, r1, r2)` | Boolean annulus mask (pixels) around the image center |
| `bscentrd(img, x, y)` | Centroid of a source near `(x, y)` |
| `fixpix(imgs, badpix, ...)` | Interpolate over bad pixels |
| `robust_sigma(data)` | Outlier-resistant standard deviation |
| `radial_profile`, `rmsmap`, `snmap`, `rms_map` | Radial profile, RMS and SNR maps |
| `fit_poly2d(z, degree)` | 2-D polynomial background fit |
| `get_region(img, rin, rout, ...)` | Annulus or sector region selection |
| `esopar(header, keyword)` | Read an ESO header keyword |

</details>

<details>
<summary><b><code>zfuncs</code>: I/O, display, helpers</b></summary>

| Function | Purpose |
| --- | --- |
| `readfits(fname, header=False)` | Read `.fits` or `.fits.Z` |
| `writefits(fname, img, header=None)` | Write FITS |
| `zdisp`, `zdisp2`, `zdisp3`, `zdispcb3` | Quick-look display of 1–3 images |
| `getfreq(img, fwhm)` | Gaussian band-pass spatial filter |
| `rmsmap_ann(img)` | Annular RMS map |
| `radprof(img, ...)` | Radial profile |
| `dist_circle(sz)` | Map of distance from the center |
| `getrgn(img, r0, r)` | Annulus mask |
| `anglediff(a1, a2)` | Angle difference (handles wrapping) |
| `keygroups`, `firstkeyvals`, `wherekeys` | Group cube frames by metadata keys (e.g. `detside`) |
| `parangle(ha, dec, lat)` | Parallactic angle |

</details>

---

## Troubleshooting

| Problem | Cause / fix |
| --- | --- |
| `dfits: command not found` | Install the ESO `dfits`/`fitsort` tools and add them to `PATH` (needed for Step 1 only). |
| `get_file_list` returns `[]` | Files must be named `SPH*.NNN.fits` (uncompressed) and be directly inside `data_dir`. For `.fits.Z` files, use `sh_funcs.get_file_list` instead, or decompress them. |
| `No calibration ... candidates found` | No matching calibration in the raw folder (wrong filter, DIT or container). Download it from the ESO archive, or write the `*list.txt` file by hand. |
| `FileNotFoundError: dark_irdis.fits` in Step 4 | `sph_cen_irdis` (Step 3) writes this file. Run Step 3 first, and make sure `skylist.txt` or `darklist.txt` exists. |
| `Centering algo failed. Check Manually.` | The satellite spots were not found. Adjust `lmaskoutw0`/`rmaskoutw0` and `lmaskinw1`/`rmaskinw1`, or check the center frame by eye. |
| One observation sorted into two folders | Folders are named by the **UT date in the file name**, so a sequence that crosses 00:00 UT is split. Merge the list files by hand. |
| Science and reference cubes look swapped | `irdis_basicreduc` assumes the science star has more frames. Swap `basicredlist1.txt` and `basicredlist2.txt`. |
| `ModuleNotFoundError: unlzw3` | `pip install unlzw3` (only needed for `.fits.Z`). |
| Notebook changes to the code are not picked up | Use `%load_ext autoreload` + `%autoreload 2`, as at the top of the notebooks. |

---

## Known limitations

- The reduction (Steps 2–7) supports **IRDIS** only. IFS is handled only during sorting.
- File names inside a dataset folder are fixed, so run **one dataset per folder**. Running a step again overwrites its outputs.
- `sph_remove_bad_images` rewrites the `.npz` file **without** the `normfac` key. Keep the `*.org.npz` backup if you need it.
- `irdis_make_flux_cube` (photometric calibration) is present, but it is not used in the default notebooks.
- Step 7 is a notebook rather than a single function, because the masks usually need checking by eye for each target.

---

## Citation

If you use this code, please cite the star-hopping method paper:

> Wahhaj, Z., et al. 2021, *A&A*, 648, A26: *"A search for a 5th planet around HR 8799 using the star-hopping RDI technique at VLT/SPHERE"*

```bibtex
@article{Wahhaj2021,
  author  = {{Wahhaj}, Z. and others},
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
