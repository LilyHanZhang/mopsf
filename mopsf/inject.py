"""
mopsf.inject
------------
Generate HEALPix injection grids and inject stpsf PSFs into mock
cal.fits exposures ready for Stage 3 + resample.

What we *do* need from the real cal.fits is:
  - the SCI header (WCS) so Stage 3 can align exposures
  - the DQ extension so outlier rejection works correctly
  - the ERR extension so drizzle can weight by inverse variance

We therefore borrow these extensions from the real cal.fits, replace
the SCI data with injected PSFs, and feed the result directly into
Stage 3.  Stage 2 is bypassed entirely.
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

log = logging.getLogger(__name__)

DEFAULT_NSIDE       = 2**12   # ~6 injection sites per NIRCam module
DEFAULT_PEAK_COUNTS = 1000.0  # arbitrary injected peak (same units as SCI)


# ── HEALPix grid ─────────────────────────────────────────────────────────────

def healpy_skycoords_in_footprint(
    wcs: WCS,
    naxis1: int,
    naxis2: int,
    nside: int = DEFAULT_NSIDE,
    edge_pad: int = 36,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        Exclude sites closer than this many pixels to the image edge,
        so that PSF cutouts don't run off the detector.

    Returns
    -------
    coords : SkyCoord
    x_pix  : ndarray of float
    y_pix  : ndarray of float
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


# ── PSF injection ─────────────────────────────────────────────────────────────

def inject_psf_at_position(
    sci: np.ndarray,
    psf: np.ndarray,
    x_cen: float,
    y_cen: float,
    peak: float = DEFAULT_PEAK_COUNTS,
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
        2-D science array (modified copy is returned; input not changed).
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
    dx = x_cen - ix
    dy = y_cen - iy

    psf_shifted = ndimage_shift(psf, shift=(dy, dx), order=3,
                                mode="constant", cval=0.0)
    psf_max = psf_shifted.max()
    if psf_max <= 0:
        return sci
    psf_scaled = psf_shifted * (peak / psf_max)

    x0, x1 = ix - hw,      ix + hw + 1
    y0, y1 = iy - hh,      iy + hh + 1
    px0     = max(0, -x0);  px1 = pw - max(0, x1 - w)
    py0     = max(0, -y0);  py1 = ph - max(0, y1 - h)
    x0      = max(0,  x0);  x1  = min(w, x1)
    y0      = max(0,  y0);  y1  = min(h, y1)

    if x1 > x0 and y1 > y0:
        sci[y0:y1, x0:x1] += psf_scaled[py0:py1, px0:px1]
    return sci


# ── mock exposure generation ──────────────────────────────────────────────────

def make_mock_exposures(
    cal_files: list[str],
    psf_cache: dict[tuple[str, str], np.ndarray],
    filter_name: str,
    out_dir: str,
    nside: int           = DEFAULT_NSIDE,
    peak_counts: float   = DEFAULT_PEAK_COUNTS,
    add_ipc: bool        = False,
) -> list[str]:
    """
    Build mock cal.fits files ready for direct input into Stage 3.

    For each real cal.fits, the SCI array is replaced with synthetic
    PSFs injected at HEALPix grid positions.  All other extensions
    (SCI header / WCS, DQ, ERR, AREA, ASDF) are copied unchanged from
    the real cal.fits so that Stage 3 astrometric alignment, outlier
    rejection, and drizzle weighting all work correctly.

    Stage 2 (flat-field, flux calibration, wisp subtraction) is
    **not** run on the mock files: those steps do not affect PSF shape
    and would be meaningless on a zero-background synthetic array.

    Parameters
    ----------
    cal_files : list of str
        Paths to real Stage 2 cal.fits files.  These supply the WCS,
        DQ, and ERR data; their SCI pixels are discarded.
    psf_cache : dict
        ``{(filter_name, detector): psf_array}`` from
        :func:`mopsf.psf_model.build_psf_cache`.
    filter_name : str
        NIRCam filter (written into the output header for provenance).
    out_dir : str
        Directory to write mock cal.fits into.
    nside : int
        HEALPix NSIDE for the injection grid.
    peak_counts : float
        Injected peak value per source (arbitrary units, same as SCI).
    add_ipc : bool
        Recorded in the output header for provenance.  Should match the
        ``add_ipc`` setting used in :func:`mopsf.psf_model.build_stpsf_psf`.

    Returns
    -------
    written : list of str
        Paths of mock cal.fits files written successfully.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    for cal_path in cal_files:
        cal_path = Path(cal_path)
        log.info("Building mock exposure from %s", cal_path.name)

        with fits.open(cal_path) as hdul:
            phdr     = hdul[0].header
            det      = phdr.get("DETECTOR", "NRCA1").upper()
            sci_hdr  = hdul["SCI"].header
            sci_data = hdul["SCI"].data.astype(np.float64)

        wcs    = WCS(sci_hdr)
        ny, nx = sci_data.shape
        key    = (filter_name, det)

        if key not in psf_cache:
            log.warning("No PSF in cache for %s — skipping %s", key, cal_path.name)
            continue

        psf_arr = psf_cache[key]
        coords, x_pos, y_pos = healpy_skycoords_in_footprint(wcs, nx, ny, nside)

        if len(x_pos) == 0:
            log.warning("No HEALPix sites in footprint of %s — skipping", cal_path.name)
            continue

        log.info("  Injecting %d sources", len(x_pos))
        mock_sci = np.zeros_like(sci_data)
        for xc, yc in zip(x_pos, y_pos):
            mock_sci = inject_psf_at_position(mock_sci, psf_arr, xc, yc, peak_counts)

        # Copy real cal.fits intact; only replace the SCI pixel array.
        # WCS, DQ, ERR, and all other extensions are kept so Stage 3
        # can align and drizzle the mock frames identically to the real data.
        with fits.open(cal_path) as out_hdul:
            out_hdul["SCI"].data = mock_sci.astype(np.float32)
            out_hdul[0].header["MPSF_INJ"] = (True,       "mPSF mock injection applied")
            out_hdul[0].header["MPSF_IPC"] = (add_ipc,    "IPC included in stpsf model")
            out_hdul[0].header["MPSF_N"]   = (len(x_pos), "Number of injected sources")
            out_hdul[0].header["MPSF_S2"]  = (False,      "Stage 2 bypassed for mock")
            out_name = out_dir / (cal_path.stem + "_mpsf.fits")
            out_hdul.writeto(out_name, overwrite=True)

        written.append(str(out_name))
        log.info("  Written → %s", out_name.name)

    return written
