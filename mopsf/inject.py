"""
mopsf.inject
------------
Generate HEALPix injection grids and inject stpsf PSFs into mock
cal.fits exposures ready for resample.

What we need from the real cal.fits:
  - SCI header  : WCS (for Stage 3 alignment) and PIXAR_A2 (pixel scale)
  - Primary header : DETECTOR, DATE-BEG (for OPD selection)
  - DQ extension : outlier rejection in Stage 3
  - ERR extension : drizzle inverse-variance weighting

The SCI data is replaced with injected PSFs; all other extensions are
copied unchanged.  Stage 2 is bypassed entirely (flat-field, flux cal,
and wisp subtraction do not affect PSF shape).

Per-site PSF computation
~~~~~~~~~~~~~~~~~~~~~~~~
Unlike a simple per-detector cache, each injection site gets its own
PSF computed at:
  - its exact detector position (field-dependent aberrations)
  - the OPD snapshot closest to DATE-BEG (wavefront drift over time)
  - the pixel scale from sqrt(PIXAR_A2) in the SCI header

This matches the approach of the JADES DR5 mPSF pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import shift as ndimage_shift
from scipy.stats import truncnorm
from .psf_model import build_stpsf_psf, pixel_scale_from_header

log = logging.getLogger(__name__)

DEFAULT_NSIDE       = 2**12   # ~6 injection sites per NIRCam module
DEFAULT_MAG_MEAN    = 20
DEFAULT_MAG_SIGMA   = 1.0
DEFAULT_MAG_LOW     = 19
DEFAULT_MAG_HIGH    = 21
DEFAULT_MAG_ZERO    = 28.0
DEFAULT_TOTAL_FLUX  = 1500.0  # arbitrary injected total flux (same units as SCI)

# ── Helper function ───────────────────────────────────────────────────────────

def truncated_norm_random(low, high, mean, sigma):
    a = (low - mean) / sigma
    b = (high - mean) / sigma
    target_mag = truncnorm.rvs(a, b, loc=mean, scale=sigma, size=1)[0]
    return target_mag

# ── HEALPix grid ──────────────────────────────────────────────────────────────

def healpy_skycoords_in_footprint(
    wcs: WCS,
    naxis1: int,
    naxis2: int,
    nside: int = DEFAULT_NSIDE,
    edge_pad: int = 36,
) -> tuple[SkyCoord, np.ndarray, np.ndarray]:
    """
    Return HEALPix pixel centres (RING scheme) inside a detector footprint.

    Parameters
    ----------
    wcs : astropy.wcs.WCS
        WCS of the detector image.
    naxis1, naxis2 : int
        Image dimensions in pixels (x, y).
    nside : int
        HEALPix NSIDE parameter.  Default 4096 ≈ 6 sites per module.
    edge_pad : int
        Exclude sites closer than this many pixels to the image edge
        so that PSF stamps don't run off the detector.

    Returns
    -------
    coords : SkyCoord
    x_pix  : ndarray of float   (image pixel x coordinates)
    y_pix  : ndarray of float   (image pixel y coordinates)
    """
    npix  = hp.nside2npix(nside)
    ipix  = np.arange(npix)
    theta, phi = hp.pix2ang(nside, ipix, lonlat=False)
    dec_hp = 90.0 - np.degrees(theta)
    ra_hp  = np.degrees(phi)

    # Bounding box from WCS corners
    c_pix = np.array([[0, 0], [naxis1, 0], [naxis1, naxis2], [0, naxis2]], dtype=float)
    skc   = wcs.pixel_to_world(c_pix[:, 0], c_pix[:, 1])
    ra_min,  ra_max  = skc.ra.deg.min(),  skc.ra.deg.max()
    dec_min, dec_max = skc.dec.deg.min(), skc.dec.deg.max()

    mask = (
        (ra_hp  >= ra_min)  & (ra_hp  <= ra_max) &
        (dec_hp >= dec_min) & (dec_hp <= dec_max)
    )
    if not mask.any():
        return SkyCoord([] * u.deg, [] * u.deg), np.array([]), np.array([])

    coords  = SkyCoord(ra=ra_hp[mask] * u.deg, dec=dec_hp[mask] * u.deg)
    x_pix, y_pix = wcs.world_to_pixel(coords)

    in_frame = (
        (x_pix >= edge_pad) & (x_pix < naxis1 - edge_pad) &
        (y_pix >= edge_pad) & (y_pix < naxis2 - edge_pad)
    )
    return coords[in_frame], x_pix[in_frame], y_pix[in_frame]

# ── PSF injection ──────────────────────────────────────────────────────────────

def inject_psf_at_position(
    sci: np.ndarray,
    psf: np.ndarray,
    x_cen: float,
    y_cen: float,
    mag_zeropoint: float = DEFAULT_MAG_ZERO,
    mag_mean: float = DEFAULT_MAG_MEAN,
    mag_sigma: float = DEFAULT_MAG_SIGMA,
    mag_low: float = DEFAULT_MAG_LOW,
    mag_high: float = DEFAULT_MAG_HIGH,
    total_flux: float | None = None,
) -> np.ndarray:
    """
    Inject a PSF stamp centred at sub-pixel position (x_cen, y_cen).

     The PSF is shifted to the correct sub-pixel phase before stamping
    so that the ePSF builder sees a well-sampled range of sub-pixel
    offsets across injection sites.

    Boundaries are handled.

    Parameters
    ----------
    sci : ndarray
        2-D science array.  A copy is returned; input is not modified.
    psf : ndarray
        2-D PSF stamp, shape (fov_pixels, fov_pixels).  Assumed normalised.
    x_cen, y_cen : float
        Sub-pixel injection centre in image pixel coordinates.
    peak : float
        Desired peak value of the injected source.

    Returns
    -------
    sci : ndarray
        Copy of the input array with the PSF added.
    """
    sci = sci.copy()
    h, w   = sci.shape
    ph, pw = psf.shape
    hh, hw = ph // 2, pw // 2

    ix, iy = int(round(x_cen)), int(round(y_cen))
    dx = x_cen - ix   # sub-pixel remainder
    dy = y_cen - iy

    psf_shifted = ndimage_shift(psf, shift=(dy, dx), order=3,
                                mode="constant", cval=0.0)
    psf_max = psf_shifted.max()
    if psf_max <= 0:
        return sci
    
    if total_flux is None:
        target_mag = truncated_norm_random(mag_low, mag_high, mag_mean, mag_sigma)
        total_flux = 10**((mag_zeropoint - target_mag) / 2.5)

    psf_scaled = psf_shifted * total_flux

    # Image bounding box
    x0, x1 = ix - hw,      ix + hw + 1
    y0, y1 = iy - hh,      iy + hh + 1
    # Corresponding PSF crop
    px0     = max(0, -x0);  px1 = pw - max(0, x1 - w)
    py0     = max(0, -y0);  py1 = ph - max(0, y1 - h)
    x0      = max(0,  x0);  x1  = min(w, x1)
    y0      = max(0,  y0);  y1  = min(h, y1)

    if x1 > x0 and y1 > y0:
        sci[y0:y1, x0:x1] += psf_scaled[py0:py1, px0:px1]
    return sci


# ── mock exposure generation ───────────────────────────────────────────────────

def make_mock_exposures(
    cal_files: list[str],
    filter_name: str,
    out_dir: str,
    nside: int           = DEFAULT_NSIDE,
    mag_mean: float = DEFAULT_MAG_MEAN,
    mag_sigma: float = DEFAULT_MAG_SIGMA,
    mag_low: float = DEFAULT_MAG_LOW,
    mag_high: float = DEFAULT_MAG_HIGH,
    default_mag_zero: float = DEFAULT_MAG_ZERO,
    total_flux:  float | None = None,
    fov_pixels: int      = 71,
    oversample: int      = 4,
    add_ipc: bool        = True,
    charge_diffusion_sigma: float | None = None,
    save_psf_dir: str | None = None,
) -> list[str]:
    """
    Build mock cal.fits files ready for direct input into Stage 3.

    For each cal.fits and each HEALPix injection site, a separate stpsf
    PSF is computed at:
      - the detector pixel position of the injection site (from WCS projection)
      - the pixel scale from ``sqrt(PIXAR_A2)`` in the SCI header
      - the OPD snapshot closest to ``DATE-BEG`` in the primary header

    The SCI array is replaced with the injected PSFs on a zero background.
    All other extensions (WCS header, DQ, ERR) are copied unchanged so
    pipeline.resample can drizzle the mock frames identically to real data.

    Parameters
    ----------
    cal_files : list of str
        Paths to real Stage 3 cal.fits files.
    filter_name : str
        NIRCam filter string (written to output header for provenance).
    out_dir : str
        Directory to write mock cal.fits files into.
    nside : int
        HEALPix NSIDE for the injection grid.
    peak_counts : float
        Injected peak value per source (same units as the SCI array).
    fov_pixels : int
        PSF stamp side length in detector pixels passed to stpsf.
    oversample : int
        Internal oversampling factor passed to stpsf.
    add_ipc : bool
        Include IPC in the stpsf model.  Set False if IPC was corrected
        in Stage 1.
    charge_diffusion_sigma : float or None
        Override charge-diffusion kernel width (arcsec).
    save_psf_dir : str or None
        If given, save each per-site PSF as a FITS file in this directory
        (useful for diagnostics).

    Returns
    -------
    written : list of str
        Paths of mock cal.fits files written successfully.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_psf_dir is not None:
        Path(save_psf_dir).mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    for cal_path in cal_files:
        cal_path = Path(cal_path)
        log.info("Building mock exposure from %s", cal_path.name)

        # ── read header metadata ───────────────────────────────────────────
        with fits.open(cal_path) as hdul:
            phdr     = hdul[0].header
            sci_hdr  = hdul["SCI"].header
            sci_data = hdul["SCI"].data.astype(np.float64)

        detector = phdr.get("DETECTOR", "NRCA1").upper()
        obs_date = phdr.get("DATE-BEG", phdr.get("DATE-OBS", "2022-01-01T00:00:00"))

        # Pixel scale from PIXAR_A2 (exact, distortion-aware)
        pixel_scale = pixel_scale_from_header(sci_hdr)
        try:
            pixel_sr = sci_hdr['PIXAR_SR']
            mag_zeropoint = -2.5 * np.log10((u.MJy / u.sr * (pixel_sr*u.sr**2) / (3631 * u.Jy)).cgs.value)
        except KeyError:
            mag_zeropoint = default_mag_zero  # fallback if PIXAR_SR is missing

        log.info(
            "  detector=%s  date=%s  pixel_scale=%.5f arcsec/px mag_zero=%.2f",
            detector, obs_date, pixel_scale, mag_zeropoint
        )

        wcs    = WCS(sci_hdr)
        ny, nx = sci_data.shape

        # ── HEALPix injection sites ────────────────────────────────────────
        coords, x_pos, y_pos = healpy_skycoords_in_footprint(
            wcs, nx, ny, nside=nside
        )
        if len(x_pos) == 0:
            log.warning(
                "No HEALPix sites in footprint of %s — skipping", cal_path.name
            )
            continue
        log.info("  %d injection sites", len(x_pos))

        # ── per-site PSF injection ─────────────────────────────────────────
        mock_sci = np.zeros_like(sci_data)

        for site_idx, (xc, yc) in enumerate(zip(x_pos, y_pos)):
            # Project sky coord → detector pixel position for stpsf
            # WCS pixel coords (0-indexed) map directly to detector coords
            det_x = float(xc)
            det_y = float(yc)

            # Optional save path for this site's PSF
            psf_save = None
            if save_psf_dir is not None:
                stem = cal_path.stem
                psf_save = (
                    Path(save_psf_dir)
                    / f"{stem}_site{site_idx:03d}_x{int(det_x)}_y{int(det_y)}.fits"
                )

            # Compute PSF for this exact position, date, and pixel scale
            psf_arr = build_stpsf_psf(
                filter_name          = filter_name,
                detector             = detector,
                detector_position    = (det_x, det_y),
                obs_date             = obs_date,
                pixel_scale          = pixel_scale,
                fov_pixels           = fov_pixels,
                oversample           = oversample,
                add_ipc              = add_ipc,
                charge_diffusion_sigma = charge_diffusion_sigma,
                save_path            = psf_save,
            )

            mock_sci = inject_psf_at_position(
                mock_sci, psf_arr, xc, yc, mag_zeropoint=mag_zeropoint, mag_mean=mag_mean,
                mag_sigma=mag_sigma, mag_low=mag_low, mag_high=mag_high, total_flux=total_flux
            )

        # ── write mock cal.fits ────────────────────────────────────────────
        # Replace SCI data only; keep WCS header, DQ, ERR, and all other
        # extensions so Stage 3 pipeline sees a valid cal.fits.
        with fits.open(cal_path) as out_hdul:
            out_hdul["SCI"].data = mock_sci.astype(np.float32)
            out_hdul[0].header["MPSF_INJ"] = (True,         "mPSF mock injection applied")
            out_hdul[0].header["MPSF_IPC"] = (add_ipc,      "IPC included in stpsf model")
            out_hdul[0].header["MPSF_N"]   = (len(x_pos),   "Number of injected sources")
            out_hdul[0].header["MPSF_S2"]  = (False,        "Stage 2 bypassed for mock")
            out_hdul[0].header["MPSF_OPD"] = (obs_date,     "OPD date used")
            out_hdul[0].header["MPSF_SCL"] = (pixel_scale,  "Pixel scale used arcsec/px")
            out_name = out_dir / (cal_path.stem + "_mpsf.fits")
            out_hdul.writeto(out_name, overwrite=True)

        written.append(str(out_name))
        log.info("  Written → %s", out_name.name)

    return written
