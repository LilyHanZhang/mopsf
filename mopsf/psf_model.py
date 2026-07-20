"""
mopsf.psf_model
---------------
Compute per-exposure, per-position optical PSFs with stpsf.

Each PSF is computed for:
  - the exact detector position of the injection site  (field-dependent
    aberrations vary across the focal plane)
  - the OPD snapshot closest to the observation date   (wavefront drift
    over the mission lifetime)
  - the pixel scale derived from the SCI header keyword PIXAR_A2

By default IPC and charge diffusion are both included (DET_DIST extension).
Set add_ipc=False if IPC has already been corrected in Stage 1.
"""

from __future__ import annotations

import warnings
import logging
from pathlib import Path

import numpy as np
import stpsf
from astropy.io import fits

log = logging.getLogger(__name__)

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_FOV_PIXELS = 71
DEFAULT_OVERSAMPLE = 4
DEFAULT_ADD_IPC    = True     # leave True unless Stage 1 corrected IPC


def pixel_scale_from_header(sci_header: fits.Header) -> float:
    """
    Derive the pixel scale in arcsec/pixel from the SCI extension header.

    Uses ``PIXAR_A2`` (pixel area in arcsec²), which is written by the
    JWST pipeline and accounts for geometric distortion at the exposure
    pointing.

    Parameters
    ----------
    sci_header : astropy.io.fits.Header
        Header of the SCI extension of a cal.fits file.

    Returns
    -------
    pixel_scale : float
        Pixel scale in arcsec/pixel.
    """
    pixar_a2 = sci_header.get("PIXAR_A2")
    if pixar_a2 is None:
        raise KeyError(
            "'PIXAR_A2' not found in SCI header. "
            "Ensure the file is a Stage 2 cal.fits produced by the JWST pipeline."
        )
    return float(np.sqrt(pixar_a2))


def build_stpsf_psf(
    filter_name: str,
    detector: str,
    detector_position: tuple[float, float],
    obs_date: str,
    pixel_scale: float,
    fov_pixels: int                      = DEFAULT_FOV_PIXELS,
    oversample: int                      = DEFAULT_OVERSAMPLE,
    add_ipc: bool                        = DEFAULT_ADD_IPC,
    charge_diffusion_sigma: float | None = None,
    save_path: str | Path | None         = None,
) -> np.ndarray:
    """
    Compute a detector-sampled stpsf PSF for one NIRCam filter, detector,
    position on the detector, and observation date.

    Parameters
    ----------
    filter_name : str
        NIRCam filter, e.g. ``"F115W"``.
    detector : str
        NIRCam detector name, e.g. ``"NRCA1"``.
    detector_position : (x, y)
        Pixel position on the detector in detector coordinates, obtained
        by projecting the injection sky position through the cal.fits WCS.
        Controls field-dependent aberrations (coma, astigmatism, etc.)
        which vary significantly across the NIRCam focal plane.
    obs_date : str
        ISO-8601 observation date, e.g. ``"2023-04-12T03:00:00"``.
        Read from the ``DATE-BEG`` keyword in the cal.fits primary header.
        Used to load the WSS OPD snapshot closest to this date so that
        wavefront drift over the mission lifetime is captured.
    pixel_scale : float
        Pixel scale in arcsec/pixel, derived from the SCI header as
        ``sqrt(PIXAR_A2)``.  Varies slightly across exposures due to
        geometric distortion.
    fov_pixels : int
        PSF stamp side length in detector pixels (should be odd).
    oversample : int
        Internal oversampling factor before rebinning to detector pixels.
    add_ipc : bool
        Include inter-pixel capacitance in the PSF model.  Set False if
        IPC was already corrected in Stage 1 ramp fitting.
    charge_diffusion_sigma : float or None
        Override charge-diffusion kernel width (arcsec).  None keeps the
        stpsf physically-motivated default for NIRCam HgCdTe detectors.
    save_path : str or Path, optional
        If given, write the PSF array to a FITS file at this path.

    Returns
    -------
    psf : ndarray, shape (fov_pixels, fov_pixels)
        Detector-sampled PSF (DET_DIST extension), normalised to sum = 1.
    """
    nrc            = stpsf.NIRCam()
    nrc.filter     = filter_name
    nrc.detector   = detector
    nrc.pixelscale = pixel_scale

    # ── field position: controls field-dependent aberrations ──────────────────
    # NIRCam coma, astigmatism, and other higher-order terms vary across
    # the focal plane; providing the exact injection-site position ensures
    # the PSF model reflects the local optical quality.
    nrc.detector_position = detector_position

    # ── OPD: wavefront snapshot closest to the observation date ───────────────
    # JWST's wavefront sensing cadence is ~2 weeks; loading the nearest
    # measurement captures slow thermal drift of the mirror segments.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nrc.load_wss_opd_by_date(obs_date, choice="closest", verbose=False)

    log.info(
        "stpsf: filter=%s  detector=%s  pos=(%.1f,%.1f)  "
        "date=%s  scale=%.5f arcsec/px  IPC=%s",
        filter_name, detector,
        detector_position[0], detector_position[1],
        obs_date, pixel_scale, add_ipc,
    )

    # ── optional detector-effect overrides ────────────────────────────────────
    if not add_ipc:
        nrc.options["add_ipc"] = False
    if charge_diffusion_sigma is not None:
        nrc.options["charge_diffusion_sigma"] = charge_diffusion_sigma

    # ── compute PSF ───────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        psf_hdul = nrc.calc_psf(
            fov_pixels = fov_pixels,
            oversample = oversample,
            normalize  = "last",
        )

    # DET_DIST includes charge diffusion + IPC (if enabled) on top of the
    # detector-sampled optical PSF — the physically correct extension.
    # DET_SAMP is the optical-only detector-sampled PSF (no detector effects).
    psf_arr = psf_hdul["DET_DIST"].data.astype(np.float64)
    psf_arr /= psf_arr.sum()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        hdr = fits.Header()
        hdr["FILTER"]   = (filter_name,             "NIRCam filter")
        hdr["DETECTOR"] = (detector,                 "NIRCam detector")
        hdr["DET_X"]    = (detector_position[0],     "Detector x position (px)")
        hdr["DET_Y"]    = (detector_position[1],     "Detector y position (px)")
        hdr["OBS_DATE"] = (obs_date,                 "Observation date for OPD")
        hdr["PIXSCALE"] = (pixel_scale,              "Pixel scale arcsec/px")
        hdr["ADD_IPC"]  = (add_ipc,                  "IPC included in model")
        hdr["COMMENT"]  = "PSF from DET_DIST extension (charge diffusion + IPC)"
        fits.writeto(save_path, psf_arr.astype(np.float32), hdr, overwrite=True)
        log.info("Saved stpsf PSF → %s", save_path)

    return psf_arr
