"""
Star selection module mimicking PSFEx's selection logic.

Replicates the FWHM-based star selection procedure from PSFEx:
1. Initial filtering by S/N, flags, ellipticity, and FWHM range
2. Compute FWHM mode using sliding window (compute_fwhmrange)
3. Apply final FWHM range based on mode
4. Return selected stars
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from astropy.visualization import simple_norm, AsinhStretch
from astropy.table import Table, Column
from astropy.stats import sigma_clipped_stats
from psfr.psfr import stack_psf
from psfr.util import oversampled2regular
from photutils.psf import fit_fwhm
from photutils.profiles import RadialProfile
from photutils.centroids import centroid_2dg
from typing import Optional, Tuple, List
from typing import Union

def compute_fwhmrange(
    fwhm: np.ndarray,
    maxvar: float = 0.5,
    minin: float = 0.0,
    maxin: float = 10.0,
) -> tuple[float, float, float]:
    """
    Compute the FWHM range associated to a series of FWHM measurements.

    Mimics PSFEx's compute_fwhmrange() in sample.c:327-372.

    Parameters
    ----------
    fwhm : np.ndarray
        Array of FWHM values (already pre-filtered by initial criteria).
    maxvar : float
        Maximum allowed FWHM variation (default 0.5, from SAMPLE_VARIABILITY).
    minin : float
        Minimum allowed FWHM (from SAMPLE_FWHMRANGE lower bound).
    maxin : float
        Maximum allowed FWHM (from SAMPLE_FWHMRANGE upper bound).

    Returns
    -------
    mode : float
        FWHM mode (center of the densest cluster).
    minout : float
        Lower bound of the final FWHM range.
    maxout : float
        Upper bound of the final FWHM range.
    """
    nfwhm = len(fwhm)
    if nfwhm == 0:
        # Default fallback (same as PSFEx)
        default_fwhm = 2.35 / (1.0 - 1.0 / 2.0)  # INTERPFAC = 2
        return default_fwhm, minin, maxin

    # Sort FWHMs
    fwhm_sorted = np.sort(fwhm)

    # Find the mode using sliding window
    nw = max(nfwhm // 4, 1)
    var = np.inf
    fmin = fwhm_sorted[0]
    dfmin = np.inf

    # Iteratively refine window size
    while var > maxvar / 2 and nw > 0:
        dfmin = np.inf
        fmin = 0.0

        # Slide window across sorted FWHM array
        for i in range(nfwhm - nw):
            df = fwhm_sorted[i + nw] - fwhm_sorted[i]
            f = (fwhm_sorted[i + nw] + fwhm_sorted[i]) / 2.0
            # Select window with smallest relative variation (df/f)
            if df * fmin < dfmin * f:
                dfmin = df
                fmin = f

        if nfwhm < 2:
            fmin = fwhm_sorted[0]
            break

        if fmin <= 0.0:
            break

        var = dfmin / fmin
        nw //= 2  # Halve window size

    # Compute final FWHM range
    dfmin_factor = np.sqrt(maxvar + 1.0)
    minout = fmin / dfmin_factor if dfmin_factor > 0.0 else 0.0
    if minout < minin:
        minout = minin
    maxout = fmin * dfmin_factor
    if maxout > maxin:
        maxout = maxin

    return fmin, minout, maxout


def select_psf_stars(
    flux_radius: np.ndarray,
    snr: np.ndarray,
    class_star: np.ndarray,
    flags: Optional[np.ndarray] = None,
    combinedflags: Optional[np.ndarray] = None,
    wflags: Optional[np.ndarray] = None,
    imaflags: Optional[np.ndarray] = None,
    elong: Optional[np.ndarray] = None,
    mag_auto: Optional[np.ndarray] = None,
    mag_bright_limit: float = 19.0,
    mag_faint_limit: float = 21.0,
    minsn: float = 10.0,
    flag_mask: int = 0x00FE,
    wflag_mask: int = 0x0000,
    imafag_mask: int = 0x0,
    max_elong: float = 1.5,
    min_starclass: float = 0.93,
    fwhm_range: tuple[float, float] = (1.0, 5.0),
    maxvar: float = 0.5,
) -> np.ndarray:
    """
    Select PSF stars mimicking PSFEx's selection procedure.

    Parameters
    ----------
    flux_radius : np.ndarray
        FLUX_RADIUS (half-light radius, r_h) for each source.
    snr : np.ndarray
        SNR_WIN (signal-to-noise ratio) for each source.
    flags : np.ndarray, optional
        SExtractor FLAGS for each source.
    wflags : np.ndarray, optional
        SExtractor FLAGS_WEIGHT for each source.
    imaflags : np.ndarray, optional
        SExtractor IMAFLAGS_ISO for each source.
    elong : np.ndarray, optional
        Elongation (A/B) for each source.
    minsn : float
        Minimum S/N for a source to be used (SAMPLE_MINSN).
    flag_mask : int
        Rejection mask on SExtractor FLAGS (SAMPLE_FLAGMASK).
    wflag_mask : int
        Rejection mask on SExtractor FLAGS_WEIGHT (SAMPLE_WFLAGMASK).
    imafag_mask : int
        Rejection mask on SExtractor IMAFLAGS_ISO (SAMPLE_IMAFLAGMASK).
    max_ellip : float
        Maximum (A-B)/(A+B) for a source to be used (SAMPLE_MAXELLIP).
    fwhm_range : tuple[float, float]
        Initial FWHM range (SAMPLE_FWHMRANGE), e.g., (1.0, 5.0).
    maxvar : float
        Allowed FWHM variability (SAMPLE_VARIABILITY), default 0.5.

    Returns
    -------
    selected_indices : np.ndarray
        Boolean array indicating which sources are selected as PSF stars.
    """

    n = len(flux_radius)
    selected = np.zeros(n, dtype=bool)

    # Compute FWHM = 2 * FLUX_RADIUS
    fwhm = 2.0 * flux_radius

    # Step 1: Initial filtering
    # S/N cut
    mask_snr = snr > minsn

    # Star class cut
    mask_class = class_star > min_starclass

    # mag cut
    mask_mag = True
    if mag_auto is not None:
        mask_mag = (mag_auto > mag_bright_limit) & (mag_auto < mag_faint_limit)

    # Flag cuts
    mask_flags = True
    if flags is not None:
        mask_flags = (flags & flag_mask) == 0
    if combinedflags is not None:
        mask_flags = ((combinedflags < 2 ) & mask_flags)

    mask_wflags = True
    if wflags is not None:
        mask_wflags = (wflags & wflag_mask) == 0

    mask_imaflags = True
    if imaflags is not None:
        mask_imaflags = (imaflags & imafag_mask) == 0

    # Ellipticity cut: (A-B)/(A+B) < max_ellip
    # elong = A/B, so (A-B)/(A+B) = (elong-1)/(elong+1)
    mask_ellip = True
    if elong is not None:
        mask_ellip = (elong < max_elong)
        #ellip = (elong - 1.0) / (elong + 1.0)
        #mask_ellip = ellip < max_ellip

    # Initial FWHM range cut
    fwhm_min_init, fwhm_max_init = fwhm_range
    mask_fwhm_init = (fwhm >= fwhm_min_init) & (fwhm < fwhm_max_init)

    # Combine all initial masks
    initial_mask = (
        mask_snr & mask_flags & mask_wflags & mask_imaflags & mask_ellip & mask_fwhm_init & mask_class & mask_mag
    )

    # Get FWHM values that passed initial filtering
    fwhm_passed = fwhm[initial_mask]

    if len(fwhm_passed) == 0:
        return selected

    # Step 2: Compute FWHM mode and final range
    mode, fwhm_min_final, fwhm_max_final = compute_fwhmrange(
        fwhm_passed, maxvar=maxvar, minin=fwhm_min_init, maxin=fwhm_max_init
    )

    # Step 3: Apply final FWHM range to initially passed sources
    mask_fwhm_final = (fwhm >= fwhm_min_final) & (fwhm < fwhm_max_final)

    # Final selection: passed initial filter AND within final FWHM range
    selected = initial_mask & mask_fwhm_final

    return selected


def select_psf_stars_from_catalog(
    catalog: dict,
    minsn: float = 10.0,
    flag_mask: int = 0x00FE,
    wflag_mask: int = 0x0000,
    imafag_mask: int = 0x0,
    max_elong: float = 1.5,
    min_starclass: float = 0.93,
    mag_bright_limit: float = 19.0,
    mag_faint_limit: float = 21.0,
    fwhm_range: tuple[float, float] = (1.0, 5.0),
    maxvar: float = 0.5,
) -> np.ndarray:
    """
    Select PSF stars from a catalog dictionary.

    Parameters
    ----------
    catalog : dict
        Dictionary containing catalog columns as numpy arrays.
        Expected keys: 'FLUX_RADIUS', 'SNR_WIN', and optionally
        'FLAGS', 'FLAGS_WEIGHT', 'IMAFLAGS_ISO', 'ELONGATION'.
    minsn : float
        Minimum S/N for a source to be used.
    flag_mask : int
        Rejection mask on SExtractor FLAGS.
    wflag_mask : int
        Rejection mask on SExtractor FLAGS_WEIGHT.
    imafag_mask : int
        Rejection mask on SExtractor IMAFLAGS_ISO.
    min_elong : float
        Maximum elongation.
    min_starclass: float
        Minimum star class.
    fwhm_range : tuple[float, float]
        Initial FWHM range.
    maxvar : float
        Allowed FWHM variability.

    Returns
    -------
    selected_indices : np.ndarray
        Boolean array indicating which sources are selected.
    """
    def _safe_array(val):
        """Convert to numpy array, returning None if input is None or missing."""
        if val is None:
            return None
        arr = np.asarray(val)
        if arr.ndim == 0 and arr.item() is None:
            return None
        return arr

    return select_psf_stars(
        flux_radius=_safe_array(catalog["flux_radius"] if "flux_radius" in catalog.colnames else None),
        snr=_safe_array(catalog["snr"] if "snr" in catalog.colnames else None),
        class_star=_safe_array(catalog["class_star"] if "class_star" in catalog.colnames else None),
        combinedflags=_safe_array(catalog["combinedflags"] if "combinedflags" in catalog.colnames else None),
        mag_auto=_safe_array(catalog["mag_auto"] if "mag_auto" in catalog.colnames else None),
        flags=_safe_array(catalog["flags"] if "flags" in catalog.colnames else None),
        wflags=_safe_array(catalog["flags_weight"] if "flags_weight" in catalog.colnames else None),
        imaflags=_safe_array(catalog["imaflags_iso"] if "imaflags_iso" in catalog.colnames else None),
        elong=_safe_array(catalog["elongation"] if "elongation" in catalog.colnames else None),
        minsn=minsn,
        flag_mask=flag_mask,
        wflag_mask=wflag_mask,
        imafag_mask=imafag_mask,
        max_elong=max_elong,
        fwhm_range=fwhm_range,
        min_starclass=min_starclass,
        maxvar=maxvar,
        mag_bright_limit=mag_bright_limit,
        mag_faint_limit=mag_faint_limit,
    )

def star_cutouts(
    sci_image: str,
    catalog_file: Union[str, Table], 
    seg_file: str,
    star_idx: np.ndarray,
    file_ext: int = 0,
    output_dir: str = "./star_cutouts",
    cutout_size: int = 71,
    save_fits: bool = True,
) -> tuple:
    if isinstance(catalog_file, str):
        outtab = ascii.read(catalog_file)
    else:
        outtab = catalog_file
    outtab1 = outtab[star_idx]
    print(f"{len(outtab1)} sources satisfy stellar criteria.")
    # --- Extract coordinates (1‑indexed) ---
    star_ids = outtab1['label']
    x_image = np.round(outtab1['xcentroid'] + 1, 3)
    y_image = np.round(outtab1['ycentroid'] + 1, 3)
    mag_auto = outtab1['mag_auto']  # 保存星等信息用于颜色编码
    # --- Create output directory ---
    os.makedirs(output_dir, exist_ok=True)
    # --- Load images ---
    sci = fits.open(sci_image)[file_ext].data
    seg = fits.open(seg_file)[0].data
    ny, nx = sci.shape
    half = cutout_size // 2
    # --- Cutout container ---
    valid_ids = []       # stars IDs that passed the boundary check
    sci_cutout_list = [] # cutout sci image data
    seg_cutout_list = [] # cutout seg image data
    mask_list = []       # boolean mask: True = use pixel (background + target star)
    valid_coords = []    # Coordinates of valid stars (cutout centers)
    valid_mags = []      # Magnitudes of valid stars for color coding
    for idx, (xc, yc, mag) in enumerate(zip(x_image, y_image, mag_auto)):
        # Round to nearest integer pixel center
        xc_int = round(xc)
        yc_int = round(yc)
        # Compute slice boundaries (left-inclusive, right-exclusive)
        xmin = xc_int - half
        xmax = xc_int + half + 1
        ymin = yc_int - half
        ymax = yc_int + half + 1        
        # Boundary check: the cutout must lie entirely within the image
        if xmin >= 0 and xmax <= nx and ymin >= 0 and ymax <= ny:
            obj_id = star_ids[idx]
            sci_cutout = sci[ymin:ymax, xmin:xmax].copy()
            seg_cutout = seg[ymin:ymax, xmin:xmax].copy()
            valid_ids.append(obj_id)
            seg_cutout_list.append(seg_cutout)
            valid_coords.append((xc, yc))
            valid_mags.append(mag)  # Save magnitude for color coding
            # mask = (seg==0) OR (seg==obj_id)  -> True = keep
            mask = ((seg_cutout == 0) | (seg_cutout == obj_id)) & (~np.isnan(sci_cutout))
            sci_cutout = np.nan_to_num(sci_cutout, nan=0.0)  # Replace NaN with 0 for stacking
            sci_cutout_list.append(sci_cutout)
            mask_list.append(mask.astype(int)) # mask area: the area to be used to stack
        else:
            print(f"Star ID {star_ids[idx]} at ({xc:.1f}, {yc:.1f}) too close to edge, skipped.")

    n_stars = len(valid_ids)
    print(f"{n_stars} out of {len(x_image)} stars are fully within the image and have been cut out.")

    # Save coordinates of valid stars only (those entirely within the image)
    if valid_coords:
        x_valid = np.array([c[0] for c in valid_coords])
        y_valid = np.array([c[1] for c in valid_coords])
        ids = np.array(valid_ids)
        np.savetxt('star_coordinates.txt', np.column_stack((x_valid, y_valid, ids)),
                   fmt=['%10.3f', '%10.3f', '%6d'])

    if n_stars == 0:
        raise RuntimeError("No valid stars found within the field of view.")

    # --- Save FITS cutouts (optional) ---
    if save_fits:
        for obj_id, star_data, seg_data, (xc, yc) in zip(
            valid_ids, sci_cutout_list, seg_cutout_list, valid_coords
        ):
            # Science cutout
            hdu = fits.PrimaryHDU(star_data)
            hdu.header['OBJID'] = (obj_id, 'Original ID')
            hdu.header['CUTX'] = (xc, 'Original x centroid (1-indexed)')
            hdu.header['CUTY'] = (yc, 'Original y centroid (1-indexed)')
            hdu.header['SIZE'] = (cutout_size, 'Cutout size in pixels')
            outname = os.path.join(output_dir, f"star{obj_id}.fits")
            hdu.writeto(outname, overwrite=True)

            # Segmentation cutout
            hdu_seg = fits.PrimaryHDU(seg_data.astype(np.int32))
            hdu_seg.header['OBJID'] = (obj_id, 'Original SExtractor ID')
            hdu_seg.header['CUTX'] = (xc, 'Original x centroid (1-indexed)')
            hdu_seg.header['CUTY'] = (yc, 'Original y centroid (1-indexed)')
            hdu_seg.header['SIZE'] = (cutout_size, 'Cutout size in pixels')
            outname_seg = os.path.join(output_dir, f"star{obj_id}_seg.fits")
            hdu_seg.writeto(outname_seg, overwrite=True)

        print(f"Cropped star images saved to {output_dir}/")
    return valid_ids, mask_list, sci_cutout_list, seg_cutout_list, valid_coords, valid_mags