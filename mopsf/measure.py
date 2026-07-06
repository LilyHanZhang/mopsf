"""
mopsf.measure
-------------
Extract the effective mosaic PSF (ePSF / mPSF) from a drizzled mock
mosaic using photutils EPSFBuilder and the known HEALPix injection
positions.

Because injection positions are exact (no centroid uncertainty), the
builder is run with recentering disabled.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp

from photutils.psf import EPSFBuilder, extract_stars
from psfr.util import oversampled2regular
from lenstronomy.Util import util, kernel_util, image_util

from .inject import healpy_skycoords_in_footprint  # re-use grid logic

log = logging.getLogger(__name__)

DEFAULT_CUTOUT_SIZE  = 65    # pixels — odd so the star is centred
DEFAULT_OVERSAMPLING = 4
DEFAULT_MAX_ITERS    = 10
DEFAULT_NSIDE        = 2**12
DEFAULT_MIN_FLUX_FRAC = 0.5  # drop stars with peak < this × median peak


# ── mosaic loading ────────────────────────────────────────────────────────────

def load_mosaic(
    mosaic_path: str,
) -> tuple[np.ndarray, np.ndarray, WCS]:
    """
    Load a drizzled mosaic FITS file.

    Handles both single-extension and multi-extension (SCI + WHT) files.

    Parameters
    ----------
    mosaic_path : str
        Path to the mosaic FITS file.

    Returns
    -------
    sci      : 2-D ndarray  (science image)
    wht      : 2-D ndarray  (weight map; 1/variance)
    wcs      : astropy WCS
    """
    with fits.open(mosaic_path) as hdul:
        ext_names = [h.name for h in hdul]
        if "SCI" in ext_names:
            sci = hdul["SCI"].data.astype(np.float64)
            wcs = WCS(hdul["SCI"].header)
            wht = (hdul["WHT"].data.astype(np.float64)
                   if "WHT" in ext_names else np.ones_like(sci))
            pixel_scale = np.sqrt(hdul["SCI"].header['PIXAR_A2']) #arcsec/pixel
        else:
            sci = hdul[0].data.astype(np.float64)
            wcs = WCS(hdul[0].header)
            wht = np.ones_like(sci)
            pixel_scale = np.sqrt(hdul[0].header['PIXAR_A2']) #arcsec/pixel

    log.info("Loaded mosaic %s — shape %s", Path(mosaic_path).name, sci.shape)
    return sci, wht, wcs, pixel_scale


def find_mosaic(mosaic_dir: str, filter_name: str) -> str:
    """
    Locate the drizzled mock mosaic produced by :func:`mopsf.pipeline.run_pipeline`.

    Parameters
    ----------
    mosaic_dir : str
        Directory to search.
    filter_name : str
        Filter name used to disambiguate multiple files in the directory.

    Returns
    -------
    path : str
        Path to the mosaic FITS file.

    Raises
    ------
    FileNotFoundError
        If no suitable file is found.
    """
    for pat in [
        os.path.join(mosaic_dir, f"*{filter_name.lower()}*mosaic_resample.fits"),
        os.path.join(mosaic_dir, f"*{filter_name.upper()}*mosaic_resample.fits"),
        os.path.join(mosaic_dir, "*.fits"),
    ]:
        hits = sorted(f for f in glob.glob(pat)
                      if "wht" not in Path(f).stem.lower())
        if hits:
            log.info("Using mosaic: %s", hits[0])
            return hits[0]
    raise FileNotFoundError(
        f"No mosaic found in {mosaic_dir} for filter {filter_name}. "
        "Run mopsf.pipeline.run_pipeline first."
    )


# ── star filtering ────────────────────────────────────────────────────────────

def _filter_low_flux(
    stars_tbl: Table,
    sci: np.ndarray,
    cutout_size: int,
    min_flux_frac: float,
) -> Table:
    """Remove injection sites where peak flux is anomalously low."""
    h   = cutout_size // 2
    ny, nx = sci.shape
    peaks = []
    for row in stars_tbl:
        ix, iy = int(round(row["x"])), int(round(row["y"]))
        y0, y1 = max(0, iy - h), min(ny, iy + h + 1)
        x0, x1 = max(0, ix - h), min(nx, ix + h + 1)
        peaks.append(float(sci[y0:y1, x0:x1].max()))

    peaks = np.array(peaks)
    stars_tbl = stars_tbl.copy()
    stars_tbl["peak"] = peaks
    threshold = min_flux_frac * np.nanmedian(peaks)
    good = peaks >= threshold
    n_bad = (~good).sum()
    if n_bad:
        log.warning(
            "Dropping %d/%d stars with peak < %.0f%% of median",
            n_bad, len(stars_tbl), min_flux_frac * 100,
        )
    return stars_tbl[good]


# ── main ePSF builder ─────────────────────────────────────────────────────────

def build_epsf(
    mosaic_path: str,
    filter_name: str,
    nside: int             = DEFAULT_NSIDE,
    cutout_size: int       = DEFAULT_CUTOUT_SIZE,
    oversampling: int      = DEFAULT_OVERSAMPLING,
    max_iters: int         = DEFAULT_MAX_ITERS,
    smoothing_kernel: str  = "quartic",
    min_flux_frac: float   = DEFAULT_MIN_FLUX_FRAC,
    save_path: str | None  = None,
    save_stars: str | None = None,
) -> tuple:
    """
    Build an effective mosaic PSF from a drizzled mock mosaic.

    Uses the known HEALPix injection positions (same grid as
    :func:`mopsf.inject.make_mock_exposures`) as input centroids, so
    no centroid refinement is needed.

    Parameters
    ----------
    mosaic_path : str
        Path to the drizzled mock mosaic FITS.
    filter_name : str
        NIRCam filter (written into the output header).
    nside : int
        HEALPix NSIDE — must match the value used in injection.
    cutout_size : int
        Side length (pixels) of each star cutout.  Must be odd.
    oversampling : int
        ePSF super-sampling factor.
    max_iters : int
        Maximum EPSFBuilder iterations.
    smoothing_kernel : str
        Smoothing kernel passed to EPSFBuilder between iterations.
    min_flux_frac : float
        Sites with peak below ``min_flux_frac × median_peak`` are
        dropped before building.
    save_path : str, optional
        If given, write the ePSF array to a FITS file here.
    save_stars : str, optional
        If given, write the star table (with peak fluxes) here.

    Returns
    -------
    epsf : photutils.psf.EPSFModel
        The fitted effective PSF model.
    fitted_stars : photutils.psf.EPSFFitters
        The collection of fitted star cutouts.
    stars_tbl : astropy.table.Table
        Table of injection sites used (with ``x``, ``y``, ``peak`` columns).
    """
    if cutout_size % 2 == 0:
        raise ValueError(f"cutout_size must be odd, got {cutout_size}")

    # ── load mosaic ──────────────────────────────────────────────────────────
    sci, wht, wcs, pixel_scale = load_mosaic(mosaic_path)
    ny, nx = sci.shape

    # ── known injection positions ────────────────────────────────────────────
    edge_pad = cutout_size // 2 + 2
    _, x_pos, y_pos = healpy_skycoords_in_footprint(
        wcs, nx, ny, nside=nside, edge_pad=edge_pad
    )

    if len(x_pos) == 0:
        raise RuntimeError(
            "No HEALPix injection sites found in the mosaic footprint. "
            "Check NSIDE, filter, and mosaic WCS."
        )

    stars_tbl = Table({"x": x_pos, "y": y_pos})

    # ── filter bad sites ─────────────────────────────────────────────────────
    stars_tbl = _filter_low_flux(stars_tbl, sci, cutout_size, min_flux_frac)
    log.info("Building ePSF from %d stars", len(stars_tbl))

    if len(stars_tbl) < 5:
        raise RuntimeError(
            f"Only {len(stars_tbl)} usable injection sites — "
            "not enough to build a reliable ePSF."
        )

    # ── extract cutouts ──────────────────────────────────────────────────────
    nddata = NDData(data=sci, mask=(wht == 0))
    stars  = extract_stars(nddata, stars_tbl, size=cutout_size)
    log.info("Extracted %d cutouts (cutout_size=%d)", len(stars), cutout_size)

    # ── build ePSF ───────────────────────────────────────────────────────────
    # recentering_func=None: positions are exact (known HEALPix centres),
    # so recentering would only inject centroid noise.
    builder = EPSFBuilder(
        oversampling         = oversampling,
        maxiters             = max_iters,
        progress_bar         = True,
        smoothing_kernel     = smoothing_kernel,
        recentering_maxiters = 0,
    )
    epsf_super, fitted_stars = builder(stars)
    epsf = kernel_util.degrade_kernel(epsf_super.data, oversampling)
    epsf = kernel_util.cut_psf(epsf, cutout_size)
    epsf = epsf / np.sum(epsf)
    log.info("ePSF built.  Shape: %s", epsf.shape)

    # ── save ePSF ────────────────────────────────────────────────────────────
    if save_path is not None:
        _save_epsf(epsf, filter_name, oversampling, cutout_size,
                   len(stars_tbl), save_path, pixel_scale=pixel_scale)

    if save_stars is not None:
        save_stars_path = Path(save_stars)
        save_stars_path.parent.mkdir(parents=True, exist_ok=True)
        stars_tbl.write(save_stars_path, overwrite=True)
        log.info("Star table → %s", save_stars_path)

    return epsf, fitted_stars, stars_tbl


def _save_epsf(epsf, filter_name: str, oversampling: int,
               cutout_size: int, n_stars: int, path: str, pixel_scale: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = fits.Header()
    hdr["FILTER"]   = (filter_name,  "NIRCam filter")
    hdr["OVERSAMP"] = (oversampling, "ePSF oversampling factor")
    hdr["CUTSIZE"]  = (cutout_size,  "Cutout size (pixels)")
    hdr["NSTARS"]   = (n_stars,      "Number of stars used")
    hdr["METHOD"]   = ("HEALPix injection + photutils EPSFBuilder", "")
    hdr["COMMENT"]  = "mPSF following JADES DR5 (arXiv:2601.15954)"
    hdr["PIXELSC"] = (pixel_scale, "Pixel scale (arcsec/pixel)")
    fits.writeto(path, epsf.astype(np.float32), hdr, overwrite=True)
    log.info("ePSF written → %s", path)
