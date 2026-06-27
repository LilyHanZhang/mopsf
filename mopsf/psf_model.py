"""
mopsf.psf_model
---------------
Compute per-exposure optical PSFs with stpsf.

By default, IPC & Charge diffusion is included.
"""

from __future__ import annotations

import warnings
import logging
from pathlib import Path

import numpy as np
import stpsf
from astropy.io import fits

log = logging.getLogger(__name__)

# ── defaults ─────────────────────────────────────────────────────────────────

DEFAULT_FOV_PIXELS  = 71
DEFAULT_PIXEL_SCALE = 0.03   # arcsec/px  (SW: ~0.031, LW: ~0.063)
DEFAULT_OVERSAMPLE  = 4
DEFAULT_ADD_IPC     = True


def build_stpsf_psf(
    filter_name: str,
    detector: str,
    fov_pixels: int        = DEFAULT_FOV_PIXELS,
    pixel_scale: float     = DEFAULT_PIXEL_SCALE,
    oversample: int        = DEFAULT_OVERSAMPLE,
    add_ipc: bool          = DEFAULT_ADD_IPC,
    charge_diffusion_sigma: float | None = None,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """
    Compute a detector-sampled stpsf PSF for one NIRCam filter + detector.

    Parameters
    ----------
    filter_name : str
        NIRCam filter, e.g. ``"F277W"``.
    detector : str
        NIRCam detector name, e.g. ``"NRCA1"``, ``"NRCB5"``.
    fov_pixels : int
        Side length of the PSF stamp in detector pixels.  Should be odd
        so the PSF centre lands on the central pixel.
    pixel_scale : float
        Output pixel scale in arcsec/pixel.
        SW channel ≈ 0.031, LW channel ≈ 0.063.
    oversample : int
        Internal oversampling factor used by stpsf before rebinning to
        detector pixels.
    add_ipc : bool
        Whether to include the inter-pixel capacitance (IPC) effect in
        the PSF model.  Set ``False`` (default) when IPC has already been
        corrected in Stage 1 ramp fitting, to avoid double-counting.
    charge_diffusion_sigma : float or None
        Override the charge-diffusion kernel width (arcsec).  ``None``
        keeps the stpsf default, which is the physically motivated value
        for NIRCam detectors.
    save_path : str or Path, optional
        If given, write the PSF array to a FITS file at this path.

    Returns
    -------
    psf : ndarray, shape (fov_pixels, fov_pixels)
        Detector-sampled PSF, normalised so it sums to 1.
    """
    nrc = stpsf.NIRCam()
    nrc.filter     = filter_name
    nrc.detector   = detector
    nrc.pixelscale = pixel_scale
    if not add_ipc:
        nrc.options["add_ipc"] = add_ipc
    if charge_diffusion_sigma is not None:
        nrc.options["charge_diffusion_sigma"] = charge_diffusion_sigma

    log.info(
        "stpsf: filter=%s  detector=%s  pixel_scale=%.4f  IPC=%s",
        filter_name, detector, pixel_scale, add_ipc,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        psf_hdul = nrc.calc_psf(
            fov_pixels = fov_pixels,
            oversample = oversample,
            normalize  = "last",
        )
    # Use the "DET_DIST" extension by default to include IPC & Charge diffusion
    # "DET_DIST" = detector-sampled (not oversampled) extension
    psf_arr = psf_hdul["DET_DIST"].data.astype(np.float64)
    psf_arr /= psf_arr.sum()   # unit-normalise

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        hdr = fits.Header()
        hdr["FILTER"]   = filter_name
        hdr["DETECTOR"] = detector
        hdr["PIXSCALE"] = pixel_scale
        hdr["ADD_IPC"]  = add_ipc
        fits.writeto(save_path, psf_arr.astype(np.float32), hdr, overwrite=True)
        log.info("Saved stpsf PSF → %s", save_path)

    return psf_arr


def build_psf_cache(
    filter_name: str,
    cal_files: list[str],
    **psf_kwargs,
) -> dict[tuple[str, str], np.ndarray]:
    """
    Build a {(filter, detector): psf_array} cache for a list of cal.fits.

    Reads the DETECTOR keyword from each file header so you don't need
    to specify detectors manually.

    Parameters
    ----------
    filter_name : str
        NIRCam filter.
    cal_files : list of str
        Paths to cal.fits files.
    **psf_kwargs
        Forwarded to :func:`build_stpsf_psf`.

    Returns
    -------
    cache : dict
        Keys are ``(filter_name, detector)`` tuples.
    """
    cache: dict[tuple[str, str], np.ndarray] = {}
    for path in cal_files:
        with fits.open(path) as hdul:
            det = hdul[0].header.get("DETECTOR", "NRCA1").upper()
        key = (filter_name, det)
        if key not in cache:
            cache[key] = build_stpsf_psf(filter_name, det, **psf_kwargs)
    return cache
