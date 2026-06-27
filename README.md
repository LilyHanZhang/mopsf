# mopsf — Mosaic PSF

Derives the effective **mosaic PSF (mPSF)** for JWST/NIRCam imaging,
following the method described in Johnson et al. (2026), JADES DR5
([arXiv:2601.15954](https://arxiv.org/abs/2601.15954), §3.4).

---

## Method

The core idea is to propagate synthetic point sources through the
*exact same mosaicing pipeline* used on the real data, so that all
drizzle-induced PSF changes are captured automatically.

1. **stpsf** computes a per-exposure optical PSF for each detector + filter.
   - IPC (`add_ipc=False`) could be disabled if it is already corrected in Stage 1 ramp fitting, since including it in the model would double-count the effect and make the PSF artificially broad.
   - IPC (`add_ipc=True`) is included by default.
   - Charge diffusion is left at the stpsf default — it is a real physical effect present in the data and is not corrected out.

2. Synthetic point sources are injected at **HEALPix grid positions**
   (NSIDE=4096, ~6 sites per NIRCam module) into zero-valued copies of
   the SCI extension of real cal.fits files.  The WCS, DQ, and ERR
   extensions are copied unchanged from the real cal.fits so that Stage 3
   alignment, outlier rejection, and drizzle weighting all work correctly.


3. The mock exposures are run through **Stage 3 + resample** with the
   same `pixfrac` used on the science data, so drizzle-induced PSF
   broadening is automatically captured.

4. **photutils EPSFBuilder** extracts the effective PSF from the
   drizzled mock mosaic using the known HEALPix injection positions —
   no centroid fitting required.

> **Limitation:** the mPSF does not capture broadening from
> inter-exposure astrometric misalignment (registration errors between
> dithers). This is an inherent limitation of the synthetic-injection
> approach noted in Johnson et al. (2026).

---

## Installation

```bash
git clone <this repo>
cd mopsf
pip install -e .
```

**Dependencies** (installed automatically):

```
numpy
astropy >= 5.3
scipy
stpsf >= 1.2.1
healpy
photutils >= 1.9
matplotlib
```

**Mosaicing pipeline** (must be installed separately):

```bash
git clone https://github.com/zezhong233/JWST-NIRCam-pipeline
# add to PYTHONPATH or install locally
```

---

## Directory layout

```
main_dir/
├── stage2/            # real _cal.fits files  ← input
├── stage3/            # real aligned files
├── mosaic/            # real mosaics
├── asn/               # association JSON files
├── wisp_templates/    # wisp templates
├── mpsf_injected/     # mock cal.fits with injected PSFs  ← Step 1 output
├── mpsf_stage3/       # aligned mock files                ← Step 2 output
├── mpsf_mosaic/       # drizzled mock mosaic              ← Step 2 output
└── mpsf_output/       # final ePSF FITS + QA plot         ← Step 3 output
```

---

## Usage

First calibrate your data using the JWST calibration pipeline (https://github.com/zezhong233/JWST-NIRCam-pipeline)

Edit `MAIN_DIR`, `FILTER`, and `PIXEL_SCALE` at the top of `run_mpsf.py`, then:

```bash
python run_mpsf.py
```

Or call individual modules from your own script:

```python
from mopsf.psf_model import build_psf_cache
from mopsf.inject    import make_mock_exposures
from mopsf.pipeline  import run_pipeline
from mopsf.measure   import build_epsf, find_mosaic

# Step 1 — build stpsf PSFs and inject into mock cal.fits
psf_cache  = build_psf_cache(filter_name="F277W", cal_files=cal_files,
                              pixel_scale=0.063, add_ipc=False)
mock_files = make_mock_exposures(cal_files, psf_cache, "F277W", out_dir="mpsf_injected/")

# Step 2 — Stage 3 + resample (Stage 2 is skipped)
run_pipeline(mock_files, filter_name="277W", ..., pixfrac=0.75)

# Step 3 — measure the effective mosaic PSF
epsf, fitted_stars, stars_tbl = build_epsf(
    mosaic_path = "mpsf_mosaic/mosaic_i2d.fits",
    filter_name = "F277W",
    save_path   = "mpsf_output/F277W_mpsf.fits",
)
```

---

## Package structure

```
mopsf/
├── mopsf/
│   ├── __init__.py      top-level imports
│   ├── psf_model.py     stpsf PSF computation
│   ├── inject.py        HEALPix grid + PSF injection into mock cal.fits
│   ├── pipeline.py      Stage 3 + resample wrapper
│   └── measure.py       photutils EPSFBuilder extraction
├── run_mpsf.py          example end-to-end driver script
└── pyproject.toml
```

---

## Reference

Johnson et al. (2026), *The JWST Advanced Deep Extragalactic Survey (JADES):
Fifth Data Release*, arXiv:2601.15954
