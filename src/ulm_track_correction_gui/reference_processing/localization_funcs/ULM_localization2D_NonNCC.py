"""2D microbubble detection, selection and sub-pixel localization (non-NCC).


Python port of ULM_localization2D.m. Formerly named ULM_localization2D; renamed
to ULM_localization2D_NonNCC when the NCC-based framework
(ULM_localization2D_NCCbased.py, ported from the lab pipeline
ULM_2D_customTracker) was added. This framework selects candidates by sorting
regional maxima of the intensity image; it does NOT use a PSF cross-correlation
(attention) map.


This function performs the detection, selection and sub-pixel localization of
bubbles on a list of input images (MatIn).


- The detection step uses regional maxima (skimage.morphology.local_maxima, the
  equivalent of MATLAB's imregionalmax with 8-connectivity).
- The selection step sorts intensities, per frame, and keeps the highest maxima.
- The localization step applies a selected sub-wavelength localization kernel
  to each selected local maximum.


DATE 2020.07.22 - VERSION 1.1
AUTHORS: Baptiste Heiles, Arthur Chavignon, Vincent Hingot. CNRS, Sorbonne
Universite, INSERM. Laboratoire d'Imagerie Biomedicale, Team PPM.
Code available under Creative Commons Attribution-NonCommercial-ShareAlike 4.0.
SPDX-License-Identifier: CC-BY-NC-SA-4.0
License: https://creativecommons.org/licenses/by-nc-sa/4.0/

Modification notice: this file is a Python adaptation of the MATLAB algorithm
identified above, integrated into this application's localization workflow.
It is not an unmodified copy of the original MATLAB implementation.
The original author credits and academic reference are retained in this header.
Redistribution and adaptation must follow the license's attribution,
noncommercial, and share-alike terms, including identifying modifications.

ACADEMIC REFERENCES TO BE CITED: Heiles, Chavignon, Hingot, Lopez, Teston and
Couture. Performance benchmarking of microbubble-localization algorithms for
ultrasound localization microscopy, Nature Biomedical Engineering, 2021.


Example
-------
>>> MatTracking = ULM_localization2D_NonNCC(MatIn, fwhm=[5, 5],
...                                         numberOfParticles=100,
...                                         LocMethod='radial')
"""


import warnings


import numpy as np
from skimage.morphology import local_maxima


from .localizeFSGradient import (
    fs_gradient_shift_maps,
    localizeFSGradientFromMaps,
)
from .localizeRadialSymmetry import localizeRadialSymmetry


#add more localization methods here
valid_loc_methods = [
    'radial',
    'fs_gradient'
]




def _matlab_round(x):
    """Round half away from zero, like MATLAB's round (numpy rounds half to even)."""
    x = float(x)
    return int(np.sign(x) * np.floor(np.abs(x) + 0.5))




def _build_localizer(method, MatIn, fs_filter_size):

    match(method):
        case "radial":
            def localize_candidate(IntensityRoi, fwhmz, fwhmx, z, x, f):
                return localizeRadialSymmetry(IntensityRoi, fwhmz, fwhmx)

            return localize_candidate


        case "fs_gradient":
            cache = {'frame': None, 'z_shift': None, 'x_shift': None}

            def localize_candidate(IntensityRoi, fwhmz, fwhmx, z, x, f):
                if cache['frame'] != f:
                    z_shift, x_shift = fs_gradient_shift_maps(
                        MatIn[:, :, f], filterSize=fs_filter_size)
                    cache['frame'] = f
                    cache['z_shift'] = z_shift
                    cache['x_shift'] = x_shift

                return localizeFSGradientFromMaps(
                    cache['z_shift'], cache['x_shift'], z, x)

            return localize_candidate

        case _:
            print(method)
            raise ValueError(
                "Wrong LocMethod selected. Supported methods are 'radial' and "
                "'FS_Gradient'."
            )




def ULM_localization2D_NonNCC(MatIn, *, fwhm, numberOfParticles,
                              LocMethod='radial', InterpMethod='spline',
                              NLocalMax=None, FSFilterSize=7):
    """Detect, select and localize microbubbles in a stack of images.


    Parameters are passed as name-value pairs (keyword arguments), exactly like
    the MATLAB version.


    Parameters
    ----------
    MatIn : array_like, shape (height, width) or (height, width, numberOfFrames)
        The sequence containing all the images.
    fwhm : (2,) sequence  (required, keyword-only)
        [fwhmx, fwhmz], bubble FWHM in pixels (usually 3 if pixel at lambda,
        5 if pixel at lambda/2).
    numberOfParticles : int  (required, keyword-only)
        Estimation of the number of particles per image.
    LocMethod : str, default 'radial'
        Localization method. Supported values are 'radial' and 'FS_Gradient'
        (case-insensitive aliases such as 'fs_gradient' are also accepted).
        Note: selecting FS_Gradient in the GUI now runs the NCC-based
        framework (ULM_localization2D_NCCbased) instead; the 'fs_gradient'
        option here is kept as legacy behavior.
    InterpMethod : str, default 'spline'
        {'bicubic', 'lanczos3', 'spline'} (used only when LocMethod == 'interp').
    NLocalMax : int or None, default None
        Max number of local maxima allowed in a ROI. None => 2 if fwhmz == 3,
        else 3 (matches the MATLAB NaN default).
    FSFilterSize : {5, 7}, default 7
        Farid-Simoncelli tap count used only when LocMethod == 'FS_Gradient'.


    Returns
    -------
    MatTracking : ndarray, shape (N, 4)
        Stores particle values and positions, one row per detected bubble:
            column 0 : intensity of the microbubble
            column 1 : super-resolved axial coordinate (z), in pixels
            column 2 : super-resolved lateral coordinate (x), in pixels
            column 3 : frame number
        Coordinates and frame numbers are 1-based (MATLAB convention), so the
        output matches ULM_localization2D.m exactly. Subtract 1 for 0-based
        Python indexing if needed.
    """
    MatIn = np.asarray(MatIn)
    orig_dtype = MatIn.dtype
    out_dtype = orig_dtype if np.issubdtype(orig_dtype, np.floating) else np.float64


    # ---- Get input data --------------------------------------------------- #
    fwhm = np.asarray(fwhm, dtype=float).ravel()
    if fwhm.size != 2:
        raise ValueError("fwhm must be a 2-element vector [fwhmx fwhmz]")
    fwhmx = float(fwhm[0])
    fwhmz = float(fwhm[1])
    numberOfParticles = int(numberOfParticles)

    loc_method = str(LocMethod).strip().lower()
    if(loc_method not in valid_loc_methods):
        print(loc_method)
        raise ValueError(
            "Wrong LocMethod selected. Supported methods are 'radial' and "
            "'FS_Gradient'."
        )


    fs_filter_size = int(FSFilterSize)


    halfz = _matlab_round(fwhmz / 2)   # half-FWHM in pixels (axial)
    halfx = _matlab_round(fwhmx / 2)   # half-FWHM in pixels (lateral)


    # Vector from -FWHM to FWHM, used for the mask's shifting.
    vectfwhmz = np.arange(-halfz, halfz + 1)
    vectfwhmx = np.arange(-halfx, halfx + 1)


    # Work with the intensity matrix; ensure a 3D (z, x, t) array.
    MatIn = np.abs(MatIn).astype(np.float64, copy=False)
    if MatIn.ndim == 2:
        MatIn = MatIn[:, :, np.newaxis]
    height, width, numberOfFrames = MatIn.shape


    # ---- Default parameters ----------------------------------------------- #
    # NLocalMax default depends on fwhm (2 if pixel at lambda, 3 if at lambda/2).
    if NLocalMax is None:
        NLocalMax = 2 if fwhmz == 3 else 3
    if loc_method == 'interp' and InterpMethod in ('bilinear', 'bicubic'):
        warnings.warn('Faster but pixelated, Weighted Average will be faster '
                      'and smoother.')


    # ---- 1. Prepare intensity matrix (crop borders) ----------------------- #
    # Build a smaller matrix to avoid the boundaries, where microbubbles cannot
    # be localized. Padding is avoided because it produces erroneous boundary
    # localizations.
    zsl = slice(halfz + 1, height - halfz - 1)   # valid rows (boundaries off)
    xsl = slice(halfx + 1, width - halfx - 1)    # valid columns (boundaries off)
    MatInReduced = np.zeros((height, width, numberOfFrames), dtype=np.float64)
    MatInReduced[zsl, xsl, :] = MatIn[zsl, xsl, :]


    # ---- 2. Detection and selection of microbubbles ----------------------- #
    # Detection of local (regional) maxima, per frame. This is equivalent to the
    # MATLAB trick of stacking frames vertically before a single imregionalmax,
    # because the cropped border rows (set to zero) separate the frames and never
    # form a regional maximum (8-connectivity, default).
    mask = np.zeros((height, width, numberOfFrames), dtype=bool)
    for f in range(numberOfFrames):
        mask[:, :, f] = local_maxima(MatInReduced[:, :, f],
                                     connectivity=2, allow_borders=True)


    IntensityMatrix = MatInReduced * mask  # intensities at regional maxima


    # Selection: keep only the numberOfParticles highest local maxima per frame.
    # Sort intensities (descending) within each frame (column-major flatten).
    flat = IntensityMatrix.reshape(height * width, numberOfFrames, order='F')
    tempMatrix = np.sort(flat, axis=0)[::-1]  # descending along rows
    # Per-frame threshold = the (numberOfParticles+1)-th highest intensity.
    thr = tempMatrix[numberOfParticles, :]
    IntensityFinal = IntensityMatrix - thr.reshape(1, 1, numberOfFrames)
    # Keep only selected maxima, carrying their original intensity value.
    MaskFinal = ((mask * IntensityFinal) > 0) * IntensityMatrix


    # Spatial and temporal coordinates of microbubbles, in column-major order to
    # match MATLAB's find / ind2sub ordering (0-based here).
    flat_idx = np.flatnonzero(MaskFinal.ravel(order='F'))
    index_mask_z, index_mask_x, index_numberOfFrames = np.unravel_index(
        flat_idx, (height, width, numberOfFrames), order='F')


    # ---- 3. Sub-wavelength localization of microbubbles ------------------- #
    Nscat = index_mask_z.size
    averageZc = np.full(Nscat, np.nan)
    averageXc = np.full(Nscat, np.nan)
    localize_candidate = _build_localizer(
        loc_method, MatIn, fs_filter_size)


    for iscat in range(Nscat):
        z = index_mask_z[iscat]
        x = index_mask_x[iscat]
        f = index_numberOfFrames[iscat]


        # 2D intensity ROI defined by fwhm, centered on the maximum.
        rows = z + vectfwhmz
        cols = x + vectfwhmx
        IntensityRoi = MatIn[:, :, f][np.ix_(rows, cols)]


        # If too many local maxima in the ROI, the bubble shape is distorted;
        # set the position to NaN (skip).
        if np.count_nonzero(local_maxima(IntensityRoi, connectivity=2,
                                         allow_borders=True)) > NLocalMax:
            continue


        # Apply the selected localization method.
        Zc, Xc = localize_candidate(IntensityRoi, fwhmz, fwhmx, z, x, f)


        # Store the super-resolved position: pixel position + sub-pixel shift.
        # (+1 converts the 0-based numpy index to MATLAB's 1-based coordinate.)
        averageZc[iscat] = Zc + (z + 1)
        averageXc[iscat] = Xc + (x + 1)


        # If the shift is larger than fwhm/2 the localization diverged: ignore.
        if abs(Zc) > fwhmz / 2 or abs(Xc) > fwhmx / 2:
            averageZc[iscat] = np.nan
            averageXc[iscat] = np.nan
            continue


    keepIndex = ~np.isnan(averageXc)


    # ---- Build MatTracking ------------------------------------------------ #
    zsel = index_mask_z[keepIndex]
    xsel = index_mask_x[keepIndex]
    fsel = index_numberOfFrames[keepIndex]


    MatTracking = np.zeros((int(np.count_nonzero(keepIndex)), 4), dtype=np.float64)
    MatTracking[:, 0] = MatInReduced[zsel, xsel, fsel]  # initial intensity
    MatTracking[:, 1] = averageZc[keepIndex]            # super-resolved z
    MatTracking[:, 2] = averageXc[keepIndex]            # super-resolved x
    MatTracking[:, 3] = fsel + 1                        # frame number (1-based)


    return MatTracking.astype(out_dtype)
