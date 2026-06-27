"""
mopsf — Mosaic PSF derivation for JWST/NIRCam imaging.

Method follows Johnson et al. (2026), JADES DR5 (arXiv:2601.15954, §3.4).

Modules
-------
psf_model   : stpsf PSF computation (IPC/charge-diffusion control)
inject      : HEALPix grid generation and PSF injection into cal.fits
pipeline    : wrapper around the mosaicing pipeline
measure     : photutils EPSFBuilder extraction from a drizzled mosaic

Quick start
-----------
>>> from mopsf.psf_model import build_stpsf_psf
>>> from mopsf.inject import make_mock_exposures
>>> from mopsf.pipeline import run_pipeline
>>> from mopsf.measure import build_epsf
"""

from .psf_model import build_stpsf_psf
from .inject import make_mock_exposures, healpy_skycoords_in_footprint
from .pipeline import run_pipeline
from .measure import build_epsf

__version__ = "0.1.0"
__all__ = [
    "build_stpsf_psf",
    "make_mock_exposures",
    "healpy_skycoords_in_footprint",
    "run_pipeline",
    "build_epsf",
]
