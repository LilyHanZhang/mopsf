"""
run_mpsf.py — example driver script
====================================

Usage
-----
    python run_mpsf.py

Edit the CONFIGURATION block below to match your data layout, then run.
"""

import glob
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from astropy.io import fits
import healpy as hp
import matplotlib.patches as patches
from matplotlib.path import Path
from astropy import wcs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import astropy.units as u
from photutils.psf import fit_fwhm
# ── configure logging (optional — mopsf uses the standard logging module) ────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── import mopsf modules ──────────────────────────────────────────────────────
from mopsf.inject    import make_mock_exposures
from mopsf.pipeline  import run_pipeline
from mopsf.measure   import build_epsf, find_mosaic, load_mosaic
# ── configure logging (optional — mopsf uses the standard logging module) ────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION — edit these paths and settings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


MAIN_DIR    = "/mnt/data/JWST/WFSS/J0100-15157/direct_image_EIGER/"
FILTER      = "F115W"
PIXEL_SCALE_DICT = {"F115W": 0.031, "F200W": 0.031, "F356W": 0.063} # arcsec/px — LW: 0.063, SW: 0.031
PIXEL_SCALE_MOSAIC_DICT = {"F115W": 0.02, "F200W": 0.02, "F356W": 0.04} # arcsec/px — LW: 0.04, SW: 0.02
MPSF_DIR    = MAIN_DIR + f'mpsf/{FILTER}'
PIXEL_SCALE = PIXEL_SCALE_DICT[FILTER]
PIXEL_SCALE_MOSAIC = PIXEL_SCALE_MOSAIC_DICT[FILTER]
# output size for mosaic; integer multiple of background box to facilitate background subtraction
RESAMPLE_OUTSHAPE_DICT = {"F115W": (21000, 12000), "F200W": (21000, 12000), "F356W": (10600, 5300)} 
# magnitude for injected stars (parameters of a truncated Gaussian distribution)
INJECT_MAG_MEAN    = 20
INJECT_MAG_SIGMA   = 1.0
INJECT_MAG_LOW     = 19
INJECT_MAG_HIGH    = 21

# Input: real Stage 3 cal.fits files
# Use tweakreg files because sky match, outlier detection etc. are not needed
CAL_FILES = sorted(glob.glob(os.path.join(MAIN_DIR, f"direct_image_{FILTER}", "stage3", f"*tweakreg.fits")))
print(len(CAL_FILES), CAL_FILES)
# Output dirs (created automatically)
# The mock cal.fits go directly into Resampling.
INJECTED_DIR = os.path.join(MPSF_DIR, "mpsf_injected")
#STAGE3_DIR   = os.path.join(MPSF_DIR, "mpsf_stage3")
MOSAIC_DIR   = os.path.join(MPSF_DIR, "mpsf_mosaic")
OUTPUT_DIR   = os.path.join(MPSF_DIR, "mpsf_output")

# Pipeline inputs (same as real-data run)
LW_DIR   = os.path.join(MPSF_DIR, "lw")
ASN_DIR  = os.path.join(MPSF_DIR, "asn")
WISP_DIR = os.path.join(MPSF_DIR, "wisp_templates")
PIXFRAC  = 0.75   # must match real-data resample

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 — build stpsf PSFs and inject into mock cal.fits
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print(f"\n{'='*60}")
print(f"  STEP 1 — PSF model + injection  [{FILTER}]")
print(f"{'='*60}\n")

# Inject into mock exposures
mock_files = make_mock_exposures(
    cal_files   = CAL_FILES,
    filter_name = FILTER,
    out_dir     = INJECTED_DIR,
    nside       = 2**12,
    mag_mean    = INJECT_MAG_MEAN,
    mag_sigma   = INJECT_MAG_SIGMA,
    mag_low     = INJECT_MAG_LOW,
    mag_high    = INJECT_MAG_HIGH,
    fov_pixels  = 135,
    add_ipc     = True,
    oversample  = 4
)
print(f"\nInjected {len(mock_files)} mock exposures → {INJECTED_DIR}\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 — run the mosaicing pipeline on mock exposures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print(f"\n{'='*60}")
print(f"  STEP 2 — mosaicing pipeline  [{FILTER}]")
print(f"{'='*60}\n")
rot_header = fits.getheader(sorted(glob.glob(os.path.join(INJECTED_DIR, f"*tweakreg_mpsf.fits")))[0], 1)
run_pipeline(
    mock_files  = mock_files,
    filter_name = FILTER.replace("F", "").lstrip("0"),  
    lw_dir      = LW_DIR,
    asn_dir     = ASN_DIR,
    wisp_dir    = WISP_DIR,
    #stage3_dir  = STAGE3_DIR,
    mosaic_dir  = MOSAIC_DIR,
    pixfrac     = PIXFRAC,
    pixel_scale_mosaic = PIXEL_SCALE_MOSAIC,
    rot_header = rot_header,
    output_shape   = RESAMPLE_OUTSHAPE_DICT[FILTER]
)
print(f"\nMock mosaic written to {MOSAIC_DIR}\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3 — measure the effective mosaic PSF (mPSF)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print(f"\n{'='*60}")
print(f"  STEP 3 — ePSF measurement  [{FILTER}]")
print(f"{'='*60}\n")

mosaic_path = find_mosaic(MOSAIC_DIR, FILTER)

os.makedirs(OUTPUT_DIR, exist_ok=True)
epsf_path  = os.path.join(OUTPUT_DIR, f"{FILTER}_mpsf.fits")
stars_path = os.path.join(OUTPUT_DIR, f"{FILTER}_mpsf_stars.fits")

epsf, fitted_stars, stars_tbl = build_epsf(
    mosaic_path  = mosaic_path,
    filter_name  = FILTER,
    cutout_size  = 135,
    oversampling = 4,
    max_iters    = 20,
    threshold    = 0.5,
    save_path    = epsf_path,
    save_stars   = stars_path,
    smoothing_kernel = "quadratic",
)

print(f"\nmPSF shape : {epsf.data.shape}")
print(f"Stars used : {len(stars_tbl)}")
print(f"ePSF saved → {epsf_path}")
print(f"Stars saved→ {stars_path}")
print("\nDone.")