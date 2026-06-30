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

# ── configure logging (optional — mopsf uses the standard logging module) ────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── import mopsf modules ──────────────────────────────────────────────────────
from mopsf.psf_model import build_psf_cache
from mopsf.inject    import make_mock_exposures
from mopsf.pipeline  import run_pipeline
from mopsf.measure   import build_epsf, find_mosaic

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION — edit these paths and settings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAIN_DIR    = "/mnt/data/JWST/WFSS/J0100-15157/direct_image_EIGER/"
MPSF_DIR    = MAIN_DIR + 'mpsf/'
FILTER      = "F115W"
PIXEL_SCALE = 0.031    # arcsec/px — LW: 0.063, SW: 0.031

# Input: real Stage 3 cal.fits files
CAL_FILES = sorted(glob.glob(os.path.join(MAIN_DIR, f"direct_image_{FILTER}", "stage3", f"*{FILTER.lower()}*_cal.fits")))

# Output dirs (created automatically)
# The mock cal.fits go directly into Resampling.
INJECTED_DIR = os.path.join(MPSF_DIR, "mpsf_injected")
STAGE3_DIR   = os.path.join(MPSF_DIR, "mpsf_stage3")
MOSAIC_DIR   = os.path.join(MPSF_DIR, "mpsf_mosaic")
OUTPUT_DIR   = os.path.join(MPSF_DIR, "mpsf_output")

# Pipeline inputs (same as real-data run)
LW_DIR   = os.path.join(MAIN_DIR, "stage2")
ASN_DIR  = os.path.join(MAIN_DIR, "asn")
WISP_DIR = os.path.join(MAIN_DIR, "wisp_templates")
PIXFRAC  = 0.75   # must match real-data resample

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 — build stpsf PSFs and inject into mock cal.fits
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print(f"\n{'='*60}")
print(f"  STEP 1 — PSF model + injection  [{FILTER}]")
print(f"{'='*60}\n")

# Build one PSF per unique detector found in the cal.fits headers
# (cached so stpsf is only called once per detector)
psf_cache = build_psf_cache(
    filter_name = FILTER,
    cal_files   = CAL_FILES,
    pixel_scale = PIXEL_SCALE,
    add_ipc     = False,     # IPC already corrected in Stage 1
    fov_pixels  = 71,
)

# Inject into mock exposures
mock_files = make_mock_exposures(
    cal_files   = CAL_FILES,
    psf_cache   = psf_cache,
    filter_name = FILTER,
    out_dir     = INJECTED_DIR,
    peak_counts = 1000.0,
)
print(f"\nInjected {len(mock_files)} mock exposures → {INJECTED_DIR}\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 — run the mosaicing pipeline on mock exposures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print(f"\n{'='*60}")
print(f"  STEP 2 — mosaicing pipeline  [{FILTER}]")
print(f"{'='*60}\n")

run_pipeline(
    mock_files  = mock_files,
    filter_name = FILTER.replace("F", "").lstrip("0"),  # pipeline expects e.g. "277W"
    lw_dir      = LW_DIR,
    asn_dir     = ASN_DIR,
    wisp_dir    = WISP_DIR,
    stage3_dir  = STAGE3_DIR,
    mosaic_dir  = MOSAIC_DIR,
    pixfrac     = PIXFRAC,
    # Stage 2 (flat, flux-cal, wisp) is skipped — see mopsf.pipeline docstring
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
    cutout_size  = 65,
    oversampling = 4,
    max_iters    = 10,
    min_flux_frac= 0.5,
    save_path    = epsf_path,
    save_stars   = stars_path,
)

print(f"\nmPSF shape : {epsf.data.shape}")
print(f"Stars used : {len(stars_tbl)}")
print(f"ePSF saved → {epsf_path}")
print(f"Stars saved→ {stars_path}")
print("\nDone.")
