# mopsf — Mosaic PSF

Derives the effective **mosaic PSF (mPSF)** for JWST/NIRCam imaging,
following the method described in Ji et al. (2024) (https://doi.org/10.3847/1538-4357/ad6e7f, Appendix A) & Johnson et al. (2026), JADES DR5 ([arXiv:2601.15954](https://arxiv.org/abs/2601.15954), §3.4).

---

## Method

The core idea is to propagate synthetic point sources through the
*exact same mosaicing pipeline* used on the real data, so that all
drizzle-induced PSF changes are captured automatically.

1. **stpsf** computes a per-exposure optical PSF for each detector + filter.
   - IPC (`add_ipc=False`) could be disabled if it is already corrected in Stage 1 ramp fitting, since including it in the model would double-count the effect and make the PSF artificially broad.
   - IPC (`add_ipc=True`) is included by default.
   - Charge diffusion is left at the stpsf default — it is not corrected out in the image calibration pipeline.
   - Detector position and optical path difference are considered.

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
git clone https://github.com/LilyHanZhang/mopsf.git
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
├── stage3/            # real _cal.fits files  ← input
├── mosaic/            # real mosaics
├── mpsf_injected/     # mock cal.fits with injected PSFs  ← Step 1 output
├── mpsf_mosaic/       # drizzled mock mosaic              ← Step 2 output
└── mpsf_output/       # final ePSF FITS + QA plot         ← Step 3 output
```

---

## Usage

First calibrate your data using the JWST calibration pipeline (https://github.com/zezhong233/JWST-NIRCam-pipeline)

You could directly use `multi_processing.py` in the pipeline: change the CRDS, Wisp template directory (downloaded at https://stsci.box.com/s/1bymvf1lkrqbdn9rnkluzqk30e8o2bne) & SCI directory to your own.

Edit `MAIN_DIR`, `FILTER`, and `PIXEL_SCALE` at the top of `run_mpsf.py`, then:

```bash
python run_mpsf.py
```

Or equivalently run `run_mpsf.ipynb` (including plotting / visualization).

Empirical PSF could be constructed using `empirical_cat.ipynb` and then `empirical_psf.ipynb`.

Comparison between stage3 empirical PSF (before drizzling) and WebbPSF could be found in `stage3_psf.ipynb`.

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

Ji, Z., Williams, C. C., Tacchella, S., et al. 2024, ApJ, 974, 135, doi: 10.3847/1538-4357/ad6e7f

Johnson et al. (2026), *The JWST Advanced Deep Extragalactic Survey (JADES):
Fifth Data Release*, arXiv:2601.15954
