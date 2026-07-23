"""
Functions to build the ldac catalog, psfex config & run psfex (multiple catlogs).
-----------------------------------
Extracted from GalfitX, with several modifications.
"""

import os
import glob
import shutil
import subprocess
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from astroquery.xmatch import XMatch
from astropy.io import fits, ascii
from astropy.visualization import simple_norm, AsinhStretch
from astropy.table import Table, Column
from astropy.stats import sigma_clipped_stats
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u
from psfr.psfr import stack_psf
from psfr.util import oversampled2regular
from photutils.psf import fit_fwhm
from photutils.profiles import RadialProfile
from photutils.centroids import centroid_2dg
from typing import Optional, Tuple, List


# ── star selection ───────────────────────────────────────────────────

def star_pre_select(
    catalog_file: str,
    mag_bright_limit: float = 19.0,
    mag_faint_limit: float = 21.0,
    crossmatch: bool = False,
    fwhm_arcsec: float = 0.13,
    elong_max: float = 1.5,
    class_star_min: float = 0.8
) -> tuple:
     
    outtab = ascii.read(catalog_file)
    outtab0 = outtab[
        (outtab['elongation'] < elong_max) &
        (outtab['class_star'] > class_star_min) &
        (outtab['combined_flags'] < 2) &
        (outtab['mag_auto'] > mag_bright_limit) &
        (outtab['mag_auto'] < mag_faint_limit) 
    ]
    print(f"{len(outtab0)} sources satisfy stellar criteria.")
    if crossmatch:
        max_distance = 1.5*fwhm_arcsec*u.arcsec
        #outtab1 = XMatch.query(cat1=outtab0, cat2='vizier:I/355/gaiadr3', 
        #                       max_distance=max_distance, colRA1='ra', colDec1='dec')  # Gaia xmatch.
        # implement when XMatch is down
        center_ra = np.median(outtab['ra'])
        center_dec = np.median(outtab['dec'])
        search_radius = np.max([np.max(outtab['ra']) - center_ra, center_ra - np.min(outtab['ra']),
                                 np.max(outtab['dec']) - center_dec, center_dec - np.min(outtab['dec'])]) * u.deg
        Vizier.ROW_LIMIT = -1
        result = Vizier.query_region(SkyCoord(ra=center_ra, dec=center_dec, unit='deg'),
                              radius=search_radius, catalog='I/355/gaiadr3')
        gaia_cat = result[0]
        print(f'search radius = {search_radius.value:.2f} deg')
        print(f'number of Gaia sources found = {len(gaia_cat)}')
        c1 = SkyCoord(ra=outtab0['ra'], dec=outtab0['dec'], unit='deg')
        c2 = SkyCoord(ra=gaia_cat['RA_ICRS'], dec=gaia_cat['DE_ICRS'], unit='deg')
        idx, sep2d, _ = c1.match_to_catalog_sky(c2)
        matched = sep2d < max_distance
        outtab1 = outtab0[matched]
        print(f"{len(outtab1)} sources after Gaia crossmatch.")
    else:
        outtab1 = outtab0
    star_id = outtab1['label']
    print(f"{len(outtab1)} sources satisfy stellar criteria.")
    return outtab1, star_id


def star_master_cat(catalog_list, star_id_list, filter_names,
                     ra_col="ra", dec_col="dec",
                     id_col="label", mag_col="mag_auto",
                     match_radius=0.1 * u.arcsec,
                     color_pair=("F115W", "F356W"), color_min=-1.7):
    """
    Restrict per-band star selections to those detected in ALL bands
    (positional cross-match), optionally further cut on a color
    (mag_blue - mag_red > color_min) if both bands in color_pair are present.

    Parameters
    ----------
    catalog_list : list of astropy Table
        One SExtractor catalog per band.
    star_id_list : list of array-like
        star_id_list[i] = the `id_col` values selected in catalog_list[i].
    filter_names : list of str
        Filter name for each entry in catalog_list, e.g. ["F090W","F115W",...].
        Must be same length/order as catalog_list.
    ra_col, dec_col : str
        Sky coordinate columns used for cross-matching.
    id_col : str
        Per-catalog identifier column (default "label").
    mag_col : str
        Magnitude column used for the color cut (default "MAG_AUTO").
    match_radius : astropy Quantity
        Max separation to call detections in different bands the same star.
    color_pair : (str, str)
        (blue_filter, red_filter) — cut applied as mag(blue) - mag(red) > color_min,
        i.e. F115W - F356W > -1.7 for the default pair. Only applied if both
        filters are present in filter_names.
    color_min : float
        Minimum allowed color (blue - red).

    Returns
    -------
    star_id_list_master : list of ndarray
        For each band, the subset of star_id_list[i] passing all criteria.
    catalog_master_list : list of astropy Table
        For each band, the corresponding subset of catalog rows.
    """
    n_bands = len(catalog_list)
    assert len(star_id_list) == n_bands
    assert len(filter_names) == n_bands

    # Subset each catalog to the selected star rows, build SkyCoords
    sel_rows, coords = [], []
    for cat, ids in zip(catalog_list, star_id_list):
        mask = np.isin(np.asarray(cat[id_col]), np.asarray(ids))
        sub = cat[mask]
        sel_rows.append(sub)
        coords.append(SkyCoord(ra=sub[ra_col], dec=sub[dec_col], unit=(u.deg, u.deg)))

    # --- Step 1: full cross-match across ALL bands ---
    # Use band 0 as the reference; a star survives only if it has a
    # match (within match_radius) in every other band.
    ref = coords[0]
    n_ref = len(ref)
    match_idx = np.full((n_bands, n_ref), -1, dtype=int)  # match_idx[j, k] = row index in band j matching ref star k
    match_idx[0] = np.arange(n_ref)
    keep_ref = np.ones(n_ref, dtype=bool)

    for j in range(1, n_bands):
        if n_ref == 0 or len(coords[j]) == 0:
            keep_ref[:] = False
            break
        idx, sep2d, _ = ref.match_to_catalog_sky(coords[j])
        good = sep2d < match_radius
        keep_ref &= good
        match_idx[j] = idx

    # Indices (into each band's sel_rows/coords) of stars present in all bands
    band_indices = [np.full(n_ref, -1, dtype=int) for _ in range(n_bands)]
    for j in range(n_bands):
        band_indices[j] = match_idx[j]

    # Apply the "detected in every band" mask
    final_keep = keep_ref.copy()

    # --- Step 2: color cut, if both filters present ---
    blue_name, red_name = color_pair
    if blue_name in filter_names and red_name in filter_names:
        b = filter_names.index(blue_name)
        r = filter_names.index(red_name)

        mag_blue = np.asarray(sel_rows[b][mag_col])[band_indices[b]]
        mag_red = np.asarray(sel_rows[r][mag_col])[band_indices[r]]
        color = mag_blue - mag_red

        color_ok = np.isfinite(color) & (color > color_min)
        final_keep &= color_ok

    # --- Build outputs ---
    star_id_list_master = []
    catalog_master_list = []
    for j in range(n_bands):
        rows = band_indices[j][final_keep]
        sub = sel_rows[j][rows]
        star_id_list_master.append(np.asarray(sub[id_col]))
        catalog_master_list.append(sub)

    return star_id_list_master, catalog_master_list

# ── build catalog & config ───────────────────────────────────────────────────

def build_input_ldac(
    sci_image: str,
    catalog_file: str,
    seg_file: str,
    output_ldac: str,
    file_ext: int = 0,
    star_id_pre: np.ndarray | None = None,
    mag_bright_limit: float = 19.0,
    mag_faint_limit: float = 21.0,
    cutout_size: int = 71,
    mask_value: float = -1e30,
    save_cutouts: bool = True,
    crossmatch: bool = False,
    cutouts_dir: str = "./star_cutouts_",
    template_file: str = "./output_assoc_temp.cat",
    fwhm_arcsec: float = 0.13,
    mag_zeropoint: float = 28.0,
    ref_pixel_scale: float = 0.074,
    star_coords_path: str = "./star_coords.txt"
) -> None:
    """
    Build a PSFEx input LDAC catalog from selected stars.

    This function reads a SExtractor catalogue, selects suitable stars,
    extracts cutouts, and creates a FITS_LDAC file (with updated header
    keywords) that can be directly used by PSFEx.

    Parameters
    ----------
    sci_image : str
        Path to the science FITS image.
    catalog_file : str
        Path to the SExtractor output catalogue (ASCII format).
    seg_file : str
        Path to the SExtractor segmentation map.
    output_ldac : str
        Path for the output PSFEx LDAC catalog.
    mag_bright_limit : float, optional (default=19.0)
        Bright magnitude limit for star selection.
    mag_faint_limit : float, optional (default=21.0)
        Faint magnitude limit for star selection.
    cutout_size : int, optional (default=71)
        Size of the square cutout (in pixels). Must match the VIGNET
        dimensions expected in the template (or be smaller if the
        template VIGNET is larger; the column definition will be
        adjusted automatically).
    mask_value : float, optional (default=-1e30)
        Value used to mask neighbouring sources in the cutout.
    save_cutouts : bool, optional (default=True)
        If True, save individual star cutouts as FITS files.
    crossmatch : bool, optional (default=False)
        If True, perform cross-matching with external catalogs (e.g., Gaia).
    cutouts_dir : str, optional (default="./star_cutouts")
        Directory where cutout FITS files will be saved (if
        `save_cutouts` is True).
    template_file : str, optional (defalut="./output_assoc_temp.cat")
        Path to an existing SExtractor ASSOC output (LDAC template).
        The template must have the same column structure and a VIGNET
        size matching `cutout_size` (or larger, though the VIGNET column
        will be rebuilt automatically).        
    fwhm_arcsec : float, optional (default=0.13)
        Approximate FWHM of stars in arcsec, used to update the
        ``SEXSFWHM`` keyword in the LDAC IMHEAD.
    mag_zeropoint : float, optional (default=28.0)
        Magnitude zero-point, used to update the ``SEXMGZPT`` keyword.
    ref_pixel_scale : float, optional (default=0.074)
        Pixel scale in arcsec/pixel, used to update the ``SEXPXSCL``
        keyword.

    Notes
    -----
    The function also updates several other SExtractor keywords
    (``NAXIS1``, ``NAXIS2``, ``SEXBKGND``, ``SEXBKDEV``,
    ``SEXTHLD``, ``SEXATHLD``, ``SEXNFIN``, ``SEXPXSCL``,
    ``SEXSFWHM``, ``SEXMGZPT``) in the LDAC IMHEAD so that they
    match the current image and star selection.

    Examples
    --------
    >>> build_input_ldac(
    ...     sci_image='sci_i.fits',
    ...     catalog_file='./sex/outcat',
    ...     seg_file='./sex/outseg.fits',
    ...     output_ldac='psfex_input_assoc.cat',
    ...     mag_bright_limit=19.0,
    ...     mag_faint_limit=21.0,
    ...     cutout_size=71,
    ...     save_cutouts=True,
    ...     cutouts_dir='./star_cutouts',
    ...     template_file='output_assoc_temp.cat',    
    ... )
    17 source satisfy stellar criteria.
    Star ID 6 at (1648.6, 34.0) too close to edge, skipped.
    ...
    14 out of 17 stars are fully within the image and have been cut out.
    Cropped star images saved to ./star_cutouts/
    LDAC 文件已生成: psfex_input_assoc.cat
    已将 NAXIS1/2 更新为 4091, 4091
    """
    # --------------------------------------------------------------------
    # 1. Read catalogue and select stars
    # --------------------------------------------------------------------
    outtab = ascii.read(catalog_file)
    if star_id_pre is not None:
        outtab1 = outtab[np.isin(outtab['label'], star_id_pre)]
    else:
        outtab1, _ = star_pre_select(catalog_file,crossmatch=crossmatch, 
                                     fwhm_arcsec=fwhm_arcsec)
    # --------------------------------------------------------------------
    # 2. Extract coordinates (1‑indexed)
    # --------------------------------------------------------------------
    star_ids = outtab1['label']
    x_image = np.round(outtab1['xcentroid'] + 1, 3) # 1-indexed
    y_image = np.round(outtab1['ycentroid'] + 1, 3)

    # --------------------------------------------------------------------
    # 3. Load images
    # --------------------------------------------------------------------
    sci = fits.open(sci_image)[file_ext].data
    seg = fits.open(seg_file)[0].data
    ny, nx = sci.shape
    half = cutout_size // 2

    # --------------------------------------------------------------------
    # 4. Extract cutouts
    # --------------------------------------------------------------------
    # Containers for valid star cutouts
    valid_ids = []         # stars IDs that passed the boundary check
    sci_cutout_list = []   # cutout sci image data
    seg_cutout_list = []   # cutout seg image data
    valid_coords = []      # Coordinates of valid stars (cutout centers)

    for idx, (xc, yc) in enumerate(zip(x_image, y_image)):
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

            # Cut out science and segmentation arrays
            sci_cutout = sci[ymin:ymax, xmin:xmax].copy()
            seg_cutout = seg[ymin:ymax, xmin:xmax].copy()

            # Mask neighbouring sources
            # Any pixel where seg_cut is not 0 AND not equal to the target ID
            # gets replaced by 'mask_value'.
            mask = ((seg_cutout != 0) & (seg_cutout != obj_id)) | (np.isnan(sci_cutout))
            sci_cutout[mask] = mask_value

            valid_ids.append(obj_id)
            sci_cutout_list.append(sci_cutout)
            seg_cutout_list.append(seg_cutout)
            valid_coords.append((xc, yc))
        else:
            print(f"Star ID {star_ids[idx]} at ({xc:.1f}, {yc:.1f}) too close to edge, skipped.")

    n_stars = len(valid_ids)
    print(f"{n_stars} out of {len(x_image)} stars are fully within the image and have been cut out.")

    # Save coordinates of valid stars only (those entirely within the image)
    if valid_coords:
        x_valid = np.array([c[0] for c in valid_coords])
        y_valid = np.array([c[1] for c in valid_coords])
        ids = np.array(valid_ids)
        np.savetxt(star_coords_path, np.column_stack((x_valid, y_valid, ids)),
                   fmt=['%10.3f', '%10.3f', '%6d'])

    if n_stars == 0:
        return None
        #raise RuntimeError("No valid stars found within the field of view.")

    # --------------------------------------------------------------------
    # 5. Save cutouts as individual FITS files (optional)
    # --------------------------------------------------------------------
    if save_cutouts:
        os.makedirs(cutouts_dir, exist_ok=True)
        for obj_id, star_data, seg_data, (xc, yc) in zip(
            valid_ids, sci_cutout_list, seg_cutout_list, valid_coords
        ):
            hdu = fits.PrimaryHDU(star_data)
            hdu.header['OBJID'] = (obj_id, 'Original  ID')
            hdu.header['CUTX'] = (xc, 'Original x centroid (1-indexed)')
            hdu.header['CUTY'] = (yc, 'Original y centroid (1-indexed)')
            hdu.header['SIZE'] = (cutout_size, 'Cutout size in pixels')
            outname = os.path.join(cutouts_dir, f"star{obj_id}.fits")
            hdu.writeto(outname, overwrite=True)

            # Segmentation cutout
            hdu_seg = fits.PrimaryHDU(seg_data.astype(np.int32))
            hdu_seg.header['OBJID'] = (obj_id, 'Original SExtractor ID')
            hdu_seg.header['CUTX'] = (xc, 'Original x centroid (1-indexed)')
            hdu_seg.header['CUTY'] = (yc, 'Original y centroid (1-indexed)')
            hdu_seg.header['SIZE'] = (cutout_size, 'Cutout size in pixels')
            outname_seg = os.path.join(cutouts_dir, f"star{obj_id}_seg.fits")
            hdu_seg.writeto(outname_seg, overwrite=True)
        print(f"Cropped star images saved to {cutouts_dir}/")

    # --------------------------------------------------------------------
    # 6. Prepare data arrays for the LDAC table
    # --------------------------------------------------------------------
    x_images = []
    y_images = []
    ra = []
    dec = []
    flux_radii = []
    elongations = []
    flags = []
    flux_apers = []      # FLUX_APER
    fluxerr_apers = []   # FLUXERR_APER

    for obj_id, sci_cutout, (xc, yc) in zip(
        valid_ids, sci_cutout_list, valid_coords
    ):
        row = outtab1[outtab1['label'] == obj_id]
        if len(row) != 1:
            raise ValueError(f"Star ID {obj_id} not found uniquely in outtab1.")
        x_images.append(xc)   # Use xc directly from valid_coords
        y_images.append(yc)   # Use yc directly from valid_coords
        ra.append(row['ra'][0])
        dec.append(row['dec'][0])
        flux_radii.append(row['flux_radius'][0])
        flags.append(row['combined_flags'][0])
        elongations.append(row['elongation'][0])
        flux_apers.append(row['kron_flux'][0])
        fluxerr_apers.append(row['kron_fluxerr'][0])  # 绝对误差，与 flux 同单位

    x_arr = np.array(x_images, dtype=np.float32)
    y_arr = np.array(y_images, dtype=np.float32)
    ra_arr = np.array(ra, dtype=np.float64)
    dec_arr = np.array(dec, dtype=np.float64)
    flux_arr = np.array(flux_radii, dtype=np.float32)
    elong_arr = np.array(elongations, dtype=np.float32)
    flags_arr = np.array(flags, dtype=np.int16)
    vign_arr = np.array(sci_cutout_list, dtype=np.float32)  # (n_stars, cutout_size, cutout_size)

    # Use actual flux measurements from SExtractor
    fap_arr = np.array(flux_apers, dtype=np.float32)         # FLUX_APER (absolute flux)
    ferr_arr = np.array(fluxerr_apers, dtype=np.float32)     # FLUXERR_APER (absolute error)
    snr_arr = fap_arr / ferr_arr                             # SNR_WIN = flux / flux_err
    print(f"SNR values: {snr_arr}")
    vect_assoc = np.array(x_images, dtype=np.float64)
    num_assoc = np.ones(n_stars, dtype=np.int32)

    # --------------------------------------------------------------------
    # 7. Copy template and modify hdu2 (LDAC_OBJECTS)
    # --------------------------------------------------------------------
    shutil.copy(template_file, output_ldac)

    with fits.open(output_ldac, mode='update') as hdul:
        hdu2 = hdul[2]

        # Build new column definitions, preserving everything except VIGNET
        new_cols = []
        for col in hdu2.columns:
            if col.name == 'VIGNET':
                n_pix = cutout_size * cutout_size
                new_format = f'{n_pix}E'
                new_col = fits.Column(name=col.name, format=new_format,
                                      unit=col.unit, disp=col.disp)
            else:
                new_col = fits.Column(name=col.name, format=col.format,
                                      unit=col.unit, disp=col.disp)
            new_cols.append(new_col)

        # Create new HDU with the correct number of rows
        new_hdu = fits.BinTableHDU.from_columns(
            fits.ColDefs(new_cols),
            header=hdu2.header,
            nrows=n_stars,
            fill=True,
            name='LDAC_OBJECTS'
        )

        # Fill with our data
        data = new_hdu.data
        data['X_IMAGE'][:]       = x_arr
        data['Y_IMAGE'][:]       = y_arr
        data['ALPHA_J2000'][:]   = ra_arr
        data['DELTA_J2000'][:]   = dec_arr
        data['FLUX_RADIUS'][:]   = flux_arr
        data['ELONGATION'][:]    = elong_arr
        data['FLAGS'][:]         = flags_arr
        data['VIGNET'][:]        = vign_arr.reshape(n_stars, -1)  # flattened
        data['FLUX_APER'][:]     = fap_arr
        data['FLUXERR_APER'][:]  = ferr_arr
        data['SNR_WIN'][:]       = snr_arr
        data['VECTOR_ASSOC'][:]  = vect_assoc
        data['NUMBER_ASSOC'][:]  = num_assoc

        # Update header keywords
        vign_idx = new_hdu.columns.names.index('VIGNET') + 1
        new_hdu.header[f'TDIM{vign_idx}'] = f'({cutout_size}, {cutout_size})'
        new_hdu.header['NAXIS2'] = n_stars

        # Recompute NAXIS1 (bytes per row)
        row_bytes = 0
        type_bytes = {'E':4, 'I':2, 'J':4, 'D':8, 'L':1, 'B':1, 'A':1, 'K':8}
        for col in new_hdu.columns:
            fmt = col.format.strip()
            if fmt[-1] in type_bytes:
                if fmt[:-1] == '':
                    repeat = 1
                else:
                    repeat = int(fmt[:-1])
                row_bytes += repeat * type_bytes[fmt[-1]]
        new_hdu.header['NAXIS1'] = row_bytes
        new_hdu.header['TFIELDS'] = len(new_hdu.columns)

        # Replace hdu2
        hdul[2] = new_hdu
        hdul.flush()

    print(f"LDAC 文件已生成: {output_ldac}")

    # --------------------------------------------------------------------
    # 8. Binary patch hdu1 (LDAC_IMHEAD) with up-to-date values
    # --------------------------------------------------------------------
    # Get global_rms from SExtractor catalog (already contains the correct value)
    global_rms = outtab['global_rms'][0]
    _, median_bkg, std_bkg = sigma_clipped_stats(sci, sigma=3.0, maxiters=5)

    with fits.open(sci_image) as sci_hdul:
        true_naxis1 = sci_hdul[file_ext].header['NAXIS1']
        true_naxis2 = sci_hdul[file_ext].header['NAXIS2']

    with open(output_ldac, 'rb') as f:
        raw = f.read()

    # Replacement pairs (old_bytes -> new_bytes) – must be exact length match
    raw = raw.replace(
        b'NAXIS1  =                 2000',
        f'NAXIS1  = {true_naxis1:20d}'.encode()
    )
    raw = raw.replace(
        b'NAXIS2  =                 2000',
        f'NAXIS2  = {true_naxis2:20d}'.encode()
    )
    raw = raw.replace(
        b'SEXBKGND=   4.254986066371E-03 / Median background level (ADU)',
        f'SEXBKGND= {median_bkg:20.10E} / Median background level (ADU)'.encode()
    )
    # Replace SEXBKDEV with global_rms from SExtractor catalog (critical for chi2 calculation)
    raw = raw.replace(
        b'SEXBKDEV=   1.305840630084E-02 / Median background RMS (ADU)',
        f'SEXBKDEV= {global_rms:20.10E} / Median background RMS (ADU)'.encode()
    )
    raw = raw.replace(
        b'SEXTHLD =   3.519492149353E-01 / Extraction threshold (ADU)',
        f'SEXTHLD = {std_bkg:20.10E} / Extraction threshold (ADU)'.encode()
    )
    raw = raw.replace(
        b'SEXATHLD=   3.519492149353E-01 / Analysis threshold (ADU)',
        f'SEXATHLD= {std_bkg:20.10E} / Analysis threshold (ADU)'.encode()
    )
    raw = raw.replace(
        b'SEXNFIN =                   17 / Final number of extracted sources',
        f'SEXNFIN = {n_stars:20d} / Final number of extracted sources'.encode()
    )
    raw = raw.replace(
        b'SEXPXSCL=   7.400000095367E-02 / Pixel scale used for measurements (arcsec)',
        f'SEXPXSCL= {ref_pixel_scale:20.10E} / Pixel scale used for measurements (arcsec)'.encode()
    )
    raw = raw.replace(
        b'SEXSFWHM=   1.299999952316E-01 / Source FWHM used for measurements (arcsec)',
        f'SEXSFWHM= {fwhm_arcsec:20.10E} / Source FWHM used for measurements (arcsec)'.encode()
    )
    raw = raw.replace(
        b'SEXMGZPT=          28.00000000 / Zero-point used for magnitudes',
        f'SEXMGZPT= {mag_zeropoint:20.8f} / Zero-point used for magnitudes'.encode()
    )
    raw = raw.replace(
        b'SEXGAIN =   1.985576562500E+04 / Gain used (e-/ADU)',
        f'SEXGAIN = {1.0:20.10E} / Gain used (e-/ADU)'.encode()
    )

    with open(output_ldac, 'wb') as f:
        f.write(raw)

    print(f"已将 NAXIS1/2 更新为 {true_naxis1}, {true_naxis2}")

def psfex_config(
    output_config: str = "config.psfex",
    *,
    psf_sampling: float = 0.5,
    psf_size: Tuple[int, int] = (141, 141),
    sample_autoselect: bool = True,
    sample_fwhmrange: str = "1, 5",
    sample_minsn: int = 10,
    psfvar_degrees: int = 0,
    outcat_name: str = "psfex_cat.txt"
    
) -> None:
    """
    Generate a PSFEx configuration file.

    Only the most commonly modified parameters are exposed; the remaining
    parameters use sensible defaults.  Comments are placed at the end of
    each line, following the PSFEx conventions.

    Parameters
    ----------
    output_config : str, optional (default "config.psfex")
        Path where the configuration file will be written.
    psf_sampling : float, optional (default 0.5)
        Sampling step in pixel units (0.0 = auto).
    psf_size : tuple[int, int], optional (default (141, 141))
        Image size of the PSF model.
    sample_autoselect : bool, optional (default True)
        Automatically select the FWHM? (Y/N)
    sample_fwhmrange : str, optional (default "1, 5")
        Allowed FWHM range for source selection.
    sample_minsn : int, optional (default 10)
        Minimum S/N for a source to be used.
    psfvar_degrees : int, optional (default 0)
        Polynomial degree for spatial PSF variation. 0 = constant PSF,
        1 = linear, 2 = quadratic. Use with `run_psfex(sample_positions=...)`.

    Examples
    --------
    >>> # Constant PSF (default)
    >>> psfex_config(
    ...     output_config='my_psfex.psfex',
    ...     psf_size=(101, 101),
    ... )
    >>> # Position-dependent PSF
    >>> psfex_config(
    ...     output_config='my_psfex.psfex',
    ...     psfvar_degrees=1,
    ... )
    Configuration written to my_psfex.psfex
    """
    autoselect = "Y" if sample_autoselect else "N"

    lines = [
        "# Default configuration file for PSFEx 3.21.1",
        "",
        "#-------------------------------- PSF model ----------------------------------",
        "",
        "BASIS_TYPE      PIXEL_AUTO           # NONE, PIXEL, GAUSS-LAGUERRE or FILE  PIXEL_AUTO",
        "BASIS_NUMBER    20              # Basis number or parameter",
        "BASIS_NAME      basis.fits      # Basis filename (FITS data-cube)",
        "BASIS_SCALE     1.0             # Gauss-Laguerre beta parameter",
        "NEWBASIS_TYPE   NONE            # Create new basis: NONE, PCA_INDEPENDENT",
        "                                # or PCA_COMMON",
        "NEWBASIS_NUMBER 8               # Number of new basis vectors",
        f"PSF_SAMPLING    {psf_sampling}             # Sampling step in pixel units (0.0 = auto)",
        "PSF_PIXELSIZE   1.0             # Effective pixel size in pixel step units",
        "PSF_ACCURACY    0.01            # Accuracy to expect from PSF \"pixel\" values",
        f"PSF_SIZE        {psf_size[0]}, {psf_size[1]}          # Image size of the PSF model",
        "PSF_RECENTER    Y               # Allow recentering of PSF-candidates Y/N ?",
        "MEF_TYPE        INDEPENDENT     # INDEPENDENT or COMMON",
        "",
        "#------------------------- Point source measurements -------------------------",
        "",
        "CENTER_KEYS     X_IMAGE,Y_IMAGE # Catalogue parameters for source pre-centering",
        "PHOTFLUX_KEY    FLUX_APER(1)    # Catalogue parameter for photometric norm.",
        "PHOTFLUXERR_KEY FLUXERR_APER(1) # Catalogue parameter for photometric error",
        "",
        "#----------------------------- PSF variability -------------------------------",
        "",
        "PSFVAR_KEYS     X_IMAGE,Y_IMAGE # Catalogue or FITS (preceded by :) params",
        "PSFVAR_GROUPS   1,1             # Group tag for each context key",
        f"PSFVAR_DEGREES  {psfvar_degrees}             # Polynom degree for each group (0=constant)",
        "PSFVAR_NSNAP    9               # Number of PSF snapshots per axis",
        "HIDDENMEF_TYPE  COMMON          # INDEPENDENT or COMMON",
        "STABILITY_TYPE  EXPOSURE        # EXPOSURE or SEQUENCE",
        "",
        "#----------------------------- Sample selection ------------------------------",
        "",
        f"SAMPLE_AUTOSELECT  {autoselect}            # Automatically select the FWHM (Y/N) ?",
        "SAMPLEVAR_TYPE     SEEING       # File-to-file PSF variability: NONE or SEEING",
        f"SAMPLE_FWHMRANGE   {sample_fwhmrange}",
        "SAMPLE_VARIABILITY 0.5         # Allowed FWHM variability (1.0 = 100%)",
        f"SAMPLE_MINSN       {sample_minsn}          # Minimum S/N for a source to be used",
        "SAMPLE_MAXELLIP    0.3          # Maximum (A-B)/(A+B) for a source to be used",
        "SAMPLE_FLAGMASK    0x00fe       # Rejection mask on SExtractor FLAGS",
        "SAMPLE_WFLAGMASK   0x0000       # Rejection mask on SExtractor FLAGS_WEIGHT",
        "SAMPLE_IMAFLAGMASK 0x0          # Rejection mask on SExtractor IMAFLAGS_ISO",
        "BADPIXEL_FILTER    N            # Filter bad-pixels in samples (Y/N) ?",
        "BADPIXEL_NMAX      0           # Maximum number of bad pixels allowed",
        "",
        "#----------------------- PSF homogeneisation kernel --------------------------",
        "",
        "HOMOBASIS_TYPE     NONE         # NONE or GAUSS-LAGUERRE",
        "HOMOBASIS_NUMBER   10           # Kernel basis number or parameter",
        "HOMOBASIS_SCALE    1.0          # GAUSS-LAGUERRE beta parameter",
        "HOMOPSF_PARAMS     2.0, 3.0     # Moffat parameters of the idealised PSF",
        "HOMOKERNEL_DIR                  # Where to write kernels (empty=same as input)",
        "HOMOKERNEL_SUFFIX  .homo   # Filename extension for homogenisation kernels",
        "",
        "#----------------------------- Output catalogs -------------------------------",
        "",
        "OUTCAT_TYPE        ASCII_HEAD         # NONE, ASCII_HEAD, ASCII, FITS_LDAC",
        f"OUTCAT_NAME        {outcat_name}           # Output catalog filename",
        "",
        "#------------------------------- Check-plots ----------------------------------",
        "",
        "CHECKPLOT_DEV       PDF         # NULL, XWIN, TK, PS, PSC, XFIG, PNG,",
        "                                # JPEG, AQT, PDF or SVG",
        "CHECKPLOT_RES       0           # Check-plot resolution (0 = default)",
        "CHECKPLOT_ANTIALIAS Y           # Anti-aliasing using convert (Y/N) ?",
        "",
        "CHECKPLOT_TYPE NONE",
        "",
        "CHECKIMAGE_TYPE NONE",
        "",
        "#------------------------------ Check-Images ---------------------------------",
        "#",
        "#CHECKIMAGE_TYPE SAMPLES,SNAPSHOTS_IMRES         # CHI,PROTOTYPES,SAMPLES,RESIDUALS,SNAPSHOTS",
        "#                                # or MOFFAT,-MOFFAT,-SYMMETRICAL",
        "#CHECKIMAGE_NAME results/diagnostics/samp.fits,   results/diagnostics/snap_imres.fits       #chi.fits,proto.fits,samp.fits,resi.fits,snap.fits",
        "#                                # Check-image filenames",
        "CHECKIMAGE_CUBE Y               # Save check-images as datacubes (Y/N) ?",
        "#",
        "",
        "#----------------------------- Miscellaneous ---------------------------------",
        "",
        "PSF_DIR                       # Where to write PSFs (empty=same as input)",
        "PSF_SUFFIX      .psf          # Filename extension for output PSF filename",
        "VERBOSE_TYPE    NORMAL          # can be QUIET,NORMAL,LOG or FULL",
        "WRITE_XML       N               # Write XML file (Y/N)?",
        "XML_NAME                        # Filename for XML output",
        "XSL_URL         file:///Users/mingyang/anaconda3/envs/py39/share/PSFEx/PSFEx.xsl",
        "                                # Filename for XSL style-sheet",
        "NTHREADS        1               # Number of simultaneous threads for",
        "                                # the SMP version of PSFEx",
        "                                # 0 = automatic",
    ]

    with open(output_config, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Configuration written to {output_config}")



def build_merged_ldac(
    ldac_files: list,
    template_file: str,
    merged_output: str,
    cutout_size: int = 71,
    mag_zeropoint: float = 28.0,
    ref_pixel_scale: float = 0.074,
    fwhm_arcsec: float = 0.13,
) -> None:
    """
    Build a single merged PSFEx input LDAC catalog by pooling star data
    already extracted (via build_input_ldac) from many exposures, and
    writing it out through the SAME template-copy + column-rebuild +
    byte-patch pipeline that build_input_ldac uses for a single exposure.
    This guarantees structural compatibility with PSFEx, since we never
    graft together already-finalized FITS_LDAC files -- we only reuse
    their star DATA.
    """
    # --------------------------------------------------------------
    # 1. Pull star data back out of each already-built exposure catalog
    # --------------------------------------------------------------
    x_list, y_list, ra_list, dec_list = [], [], [], []
    flux_radius_list, elong_list, flags_list = [], [], []
    vign_list, fap_list, ferr_list = [], [], []

    for f in ldac_files:
        with fits.open(f) as hdul:
            d = hdul[2].data
            vign_idx = hdul[2].columns.names.index('VIGNET') + 1
            tdim = hdul[2].header.get(f'TDIM{vign_idx}')
            expected = f'({cutout_size}, {cutout_size})'
            if tdim != expected:
                raise ValueError(f"{f}: VIGNET TDIM {tdim} != expected {expected}")

            n = len(d)
            x_list.append(np.array(d['X_IMAGE']))
            y_list.append(np.array(d['Y_IMAGE']))
            ra_list.append(np.array(d['ALPHA_J2000']))
            dec_list.append(np.array(d['DELTA_J2000']))
            flux_radius_list.append(np.array(d['FLUX_RADIUS']))
            elong_list.append(np.array(d['ELONGATION']))
            flags_list.append(np.array(d['FLAGS']))
            # reshape VIGNET back to (n, cutout_size, cutout_size)
            vign_list.append(np.array(d['VIGNET']).reshape(n, cutout_size, cutout_size))
            fap_list.append(np.array(d['FLUX_APER']))
            ferr_list.append(np.array(d['FLUXERR_APER']))

    x_arr = np.concatenate(x_list).astype(np.float32)
    y_arr = np.concatenate(y_list).astype(np.float32)
    ra_arr = np.concatenate(ra_list).astype(np.float64)
    dec_arr = np.concatenate(dec_list).astype(np.float64)
    flux_arr = np.concatenate(flux_radius_list).astype(np.float32)
    elong_arr = np.concatenate(elong_list).astype(np.float32)
    flags_arr = np.concatenate(flags_list).astype(np.int16)
    vign_arr = np.concatenate(vign_list).astype(np.float32)  # (n_total, cutout_size, cutout_size)
    fap_arr = np.concatenate(fap_list).astype(np.float32)
    ferr_arr = np.concatenate(ferr_list).astype(np.float32)
    snr_arr = fap_arr / ferr_arr
    n_total = len(x_arr)
    vect_assoc = np.array(x_arr, dtype=np.float64)
    num_assoc = np.ones(n_total, dtype=np.int32)

    print(f"Pooled {n_total} stars from {len(ldac_files)} exposures.")

    # --------------------------------------------------------------
    # 2. Same as build_input_ldac step 7: fresh template copy + rebuild hdu2
    # --------------------------------------------------------------
    shutil.copy(template_file, merged_output)

    with fits.open(merged_output, mode='update') as hdul:
        hdu2 = hdul[2]
        new_cols = []
        for col in hdu2.columns:
            if col.name == 'VIGNET':
                n_pix = cutout_size * cutout_size
                new_col = fits.Column(name=col.name, format=f'{n_pix}E',
                                      unit=col.unit, disp=col.disp)
            else:
                new_col = fits.Column(name=col.name, format=col.format,
                                      unit=col.unit, disp=col.disp)
            new_cols.append(new_col)

        new_hdu = fits.BinTableHDU.from_columns(
            fits.ColDefs(new_cols), header=hdu2.header,
            nrows=n_total, fill=True, name='LDAC_OBJECTS'
        )

        data = new_hdu.data
        data['X_IMAGE'][:]      = x_arr
        data['Y_IMAGE'][:]      = y_arr
        data['ALPHA_J2000'][:]  = ra_arr
        data['DELTA_J2000'][:]  = dec_arr
        data['FLUX_RADIUS'][:]  = flux_arr
        data['ELONGATION'][:]   = elong_arr
        data['FLAGS'][:]        = flags_arr
        data['VIGNET'][:]       = vign_arr.reshape(n_total, -1)
        data['FLUX_APER'][:]    = fap_arr
        data['FLUXERR_APER'][:] = ferr_arr
        data['SNR_WIN'][:]      = snr_arr
        data['VECTOR_ASSOC'][:] = vect_assoc
        data['NUMBER_ASSOC'][:] = num_assoc

        vign_idx = new_hdu.columns.names.index('VIGNET') + 1
        new_hdu.header[f'TDIM{vign_idx}'] = f'({cutout_size}, {cutout_size})'
        new_hdu.header['NAXIS2'] = n_total

        row_bytes = 0
        type_bytes = {'E':4, 'I':2, 'J':4, 'D':8, 'L':1, 'B':1, 'A':1, 'K':8}
        for col in new_hdu.columns:
            fmt = col.format.strip()
            repeat = 1 if fmt[:-1] == '' else int(fmt[:-1])
            row_bytes += repeat * type_bytes[fmt[-1]]
        new_hdu.header['NAXIS1'] = row_bytes
        new_hdu.header['TFIELDS'] = len(new_hdu.columns)

        hdul[2] = new_hdu
        hdul.flush()

    print(f"Merged LDAC written to {merged_output}")

    # --------------------------------------------------------------
    # 3. Same as build_input_ldac step 8: raw byte-patch hdu1 (LDAC_IMHEAD)
    #    Use aggregate/representative values since this is now pooled
    #    across many exposures.
    # --------------------------------------------------------------
    max_naxis = 8192  # safely large; only needs to bound merged X/Y_IMAGE values,
                       # not represent a physically meaningful single image
    with open(merged_output, 'rb') as f:
        raw = f.read()

    raw = raw.replace(
        b'NAXIS1  =                 2000',
        f'NAXIS1  = {max_naxis:20d}'.encode()
    )
    raw = raw.replace(
        b'NAXIS2  =                 2000',
        f'NAXIS2  = {max_naxis:20d}'.encode()
    )
    raw = raw.replace(
        b'SEXNFIN =                   17 / Final number of extracted sources',
        f'SEXNFIN = {n_total:20d} / Final number of extracted sources'.encode()
    )
    raw = raw.replace(
        b'SEXPXSCL=   7.400000095367E-02 / Pixel scale used for measurements (arcsec)',
        f'SEXPXSCL= {ref_pixel_scale:20.10E} / Pixel scale used for measurements (arcsec)'.encode()
    )
    raw = raw.replace(
        b'SEXSFWHM=   1.299999952316E-01 / Source FWHM used for measurements (arcsec)',
        f'SEXSFWHM= {fwhm_arcsec:20.10E} / Source FWHM used for measurements (arcsec)'.encode()
    )
    raw = raw.replace(
        b'SEXMGZPT=          28.00000000 / Zero-point used for magnitudes',
        f'SEXMGZPT= {mag_zeropoint:20.8f} / Zero-point used for magnitudes'.encode()
    )
    raw = raw.replace(
        b'SEXGAIN =   1.985576562500E+04 / Gain used (e-/ADU)',
        f'SEXGAIN = {1.0:20.10E} / Gain used (e-/ADU)'.encode()
    )

    with open(merged_output, 'wb') as f:
        f.write(raw)

    print(f"Merged catalog finalized: {n_total} stars, NAXIS1/2 set to {max_naxis}")



def run_psfex_multi(
    catalog_list,
    config,
    output_dir="./psf_results",
    listfile="psfex_catalogs.list",
):
    """
    Run PSFEx on many catalogs at once via an @listfile, as recommended
    by the PSFEx docs when the number of catalogs is too large for the
    shell command line.

    NOTE: per PSFEx's documented default behavior (STABILITY_TYPE
    EXPOSURE), this may produce one .psf file PER INPUT CATALOG rather
    than a single pooled model, even with STABILITY_TYPE SEQUENCE set
    in the config (the docs do not clearly specify SEQUENCE's exact
    output behavior). Check len(glob.glob(output_dir + '/*.psf')) after
    running -- if it equals len(catalog_list), you got per-exposure
    models, not a pooled one, and should use the manual-merge approach
    instead.
    """
    if not os.path.isfile(config):
        raise FileNotFoundError(f"Config not found: {config}")
    for f in catalog_list:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Catalog not found: {f}")

    os.makedirs(output_dir, exist_ok=True)

    # Write the @listfile: one catalog path per line
    with open(listfile, "w") as f:
        f.write("\n".join(catalog_list) + "\n")

    cmd = ["psfex", f"@{listfile}", "-c", config, "-PSF_DIR", output_dir]
    print("Running:", " ".join(cmd))
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"PSFEx exited with error code {ret.returncode}")

    psf_files = sorted(glob.glob(os.path.join(output_dir, "*.psf")))
    print(f"PSFEx produced {len(psf_files)} .psf file(s) "
          f"from {len(catalog_list)} input catalogs.")
    if len(psf_files) == 1:
        print(f"Single pooled PSF model: {psf_files[0]}")
    elif len(psf_files) == len(catalog_list):
        print("WARNING: got one PSF per catalog, not a single pooled "
              "model. STABILITY_TYPE SEQUENCE did not merge statistics "
              "into one output as hoped -- use the manual LDAC-merge "
              "approach instead for a guaranteed single combined PSF.")
    return psf_files