"""
mopsf.pipeline
--------------
Run Stage 3 (alignment + outlier rejection) and resample (drizzle) on
the mock cal.fits files produced by :mod:`mopsf.inject`.

Stage 2 is intentionally skipped
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The mock cal.fits files already contain:
  - Synthetic PSF-only SCI data on a zero background
  - Real WCS, DQ, and ERR copied from the genuine cal.fits

There is no real signal to flat-field or flux-calibrate, and no wisp
artefacts to subtract.  Running Stage 2 on these files would be a no-op
at best and could corrupt the zero background at worst.

We therefore feed the mock files directly into Stage 3
(TweakReg / OutlierDetection) and then resample with the same pixfrac
used on the science data.
"""

from __future__ import annotations

import glob
import importlib
import logging
import os
import shutil
from pathlib import Path
import sys

log = logging.getLogger(__name__)
path = Path('~').expanduser() #the path where JWST-NIRCam-pipeline is located, change if needed
sys.path.append(str(path))

def run_pipeline(
    mock_files: list[str],
    filter_name: str,
    lw_dir: str,
    asn_dir: str,
    wisp_dir: str,
    stage3_dir: str,
    mosaic_dir: str,
    pixfrac: float = 0.75,
) -> str:
    """
    Run resample (drizzle) on
    mock-injected cal.fits files.

    Stage 2 & 3 are **not** run; see module docstring for rationale.

    Parameters
    ----------
    mock_files : list of str
        Paths to mock cal.fits from :func:`mopsf.inject.make_mock_exposures`.
    filter_name : str
        NIRCam filter string accepted by the pipeline (e.g. ``"277W"``).
    lw_dir : str
        Directory of real long-wave cal.fits (required by pipeline init).
    asn_dir : str
        Directory of association JSON files.
    wisp_dir : str
        Directory of wisp template files (required by pipeline init,
        not used for mock frames).
    stage3_dir : str
        Working directory for Stage 3 outputs (aligned, outlier-rejected).
    mosaic_dir : str
        Output directory for the drizzled mock mosaic.
    pixfrac : float
        Drizzle pixfrac.  **Must match the real-data resample step exactly.**

    Returns
    -------
    mosaic_dir : str
        Directory containing the drizzled mock mosaic.

    Raises
    ------
    ImportError
        If the ``pipeline`` package is not importable.
    RuntimeError
        If no output mosaic FITS is found after the run.
    """
    try:
        #from JWST-NIRCam-pipeline.pipeline import pipeline as _Pipeline
        pipeline_module = importlib.import_module('JWST-NIRCam-pipeline.pipeline')
        _Pipeline = pipeline_module.pipeline
    except ImportError as exc:
        raise ImportError(
            "Could not import 'pipeline'.  "
            "Clone https://github.com/zezhong233/JWST-NIRCam-pipeline "
            "and add it to PYTHONPATH."
        ) from exc

    for d in (stage3_dir, mosaic_dir):
        Path(d).mkdir(parents=True, exist_ok=True)        
    # The pipeline expects its input cal.fits in stage2_dir.
    # We use stage3_dir as a staging area so we don't mix mock and real files.

    staged_cal_dir = Path(mosaic_dir) / "staged_cal"
    staged_cal_dir.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    for src in mock_files:
        dst = staged_cal_dir / Path(src).name
        shutil.copy2(src, dst)
        staged.append(str(dst))
    log.info("Staged %d mock cal.fits → %s", len(staged), staged_cal_dir)

    pl = _Pipeline(
        lw_dir    = lw_dir,
        asn_dir   = asn_dir,
        wisp_dir  = wisp_dir,
        #stage0_dir= str(staged_cal_dir),  # unused
        #stage1_dir= str(staged_cal_dir),  # unused
        #stage2_dir= str(staged_cal_dir),  # mock cal.fits live here
        stage3_dir= str(staged_cal_dir),
        mosaic_dir= mosaic_dir,
        filter    = filter_name,
    )

    # ── Stage 3: astrometric alignment + outlier rejection ────────────────────
    # This reads from stage2_dir (our staged mock cal.fits) and writes
    # aligned, outlier-flagged files to stage3_dir.
    #log.info("Stage 3 (alignment + outlier rejection) on %d mock files …", len(staged))
    #pl.stage3_part3()

    # ── Resample: drizzle with same pixfrac as real data ──────────────────────
    log.info("Resample (drizzle) with pixfrac=%.2f …", pixfrac)
    pl.resample(pixfrac=pixfrac)

    # Verify output
    mosaics = [
        f for f in glob.glob(os.path.join(mosaic_dir, "*.fits"))
        if "wht" not in Path(f).stem.lower()
    ]
    if not mosaics:
        raise RuntimeError(
            f"Pipeline finished but no mosaic FITS found in {mosaic_dir}."
        )
    log.info("Mock mosaic ready in %s", mosaic_dir)
    return mosaic_dir
