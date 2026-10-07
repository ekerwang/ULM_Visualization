"""Farid-Simoncelli gradient sub-pixel localization helpers.

This module ports the 2D FS_Gradient localization kernel used by the MATLAB
custom tracker. The derivative maps are computed on the full frame with
``conv2(..., 'same')`` semantics, then sampled at integer peak locations.
"""

import numpy as np
from scipy.signal import convolve2d


_K5 = np.array([0.030320, 0.249724, 0.439911, 0.249724, 0.030320],
               dtype=np.float64)
_D1_5 = np.array([0.104550, 0.292315, 0.000000, -0.292315, -0.104550],
                 dtype=np.float64)
_D2_5 = np.array([0.232905, 0.002668, -0.471147, 0.002668, 0.232905],
                 dtype=np.float64)

_K7 = np.array([0.004711, 0.069321, 0.245410, 0.361117,
                0.245410, 0.069321, 0.004711], dtype=np.float64)
_D1_7 = np.array([0.018708, 0.125376, 0.193091, 0.000000,
                  -0.193091, -0.125376, -0.018708], dtype=np.float64)
_D2_7 = np.array([0.055336, 0.137778, -0.056554, -0.273118,
                  -0.056554, 0.137778, 0.055336], dtype=np.float64)


def _filter_taps(filterSize):
    filterSize = int(filterSize)
    if filterSize == 5:
        return _K5, _D1_5, _D2_5
    if filterSize == 7:
        return _K7, _D1_7, _D2_7
    raise ValueError(f"FS_Gradient filterSize {filterSize} is not supported")


def _conv2_vectors(image, hcol, hrow):
    """Convolve with MATLAB conv2(hcol, hrow, image, 'same') semantics."""
    kernel = np.outer(hcol, hrow)
    return convolve2d(image, kernel, mode='same', boundary='fill',
                      fillvalue=0)


def fs_gradient_shift_maps(I, filterSize=7):
    """Compute axial and lateral FS-gradient sub-pixel shift maps.

    Parameters
    ----------
    I : array_like, shape (Nz, Nx)
        2D frame data.
    filterSize : {5, 7}, default 7
        Farid-Simoncelli tap count, matching the MATLAB implementation.

    Returns
    -------
    z_shift, x_shift : ndarray
        Pixel shifts at each integer pixel. ``z_shift`` is axial/row shift and
        ``x_shift`` is lateral/column shift.
    """
    I = np.asarray(I, dtype=np.float64)
    k, d1, d2 = _filter_taps(filterSize)

    gx = _conv2_vectors(I, k, d1)
    gxx = _conv2_vectors(I, k, d2)

    gy = _conv2_vectors(I, d1, k)
    gyy = _conv2_vectors(I, d2, k)

    with np.errstate(divide='ignore', invalid='ignore'):
        x_shift = -gx / gxx
        z_shift = -gy / gyy

    x_shift[np.isinf(x_shift)] = 0.0
    z_shift[np.isinf(z_shift)] = 0.0

    return z_shift, x_shift


def localizeFSGradientFromMaps(z_shift, x_shift, z, x):
    """Sample precomputed FS-gradient maps at one integer peak location."""
    Zc = float(z_shift[z, x])
    Xc = float(x_shift[z, x])

    if np.isinf(Zc):
        Zc = 0.0
    if np.isinf(Xc):
        Xc = 0.0
    if np.isnan(Zc) or np.isnan(Xc):
        return np.nan, np.nan

    return Zc, Xc


def localizeFSGradient(I, fwhmz=None, fwhmx=None, filterSize=7):
    """Compute the FS-gradient shift at the center of a 2D ROI.

    ``fwhmz`` and ``fwhmx`` are accepted for signature compatibility with other
    localizers; they are not used by the FS-gradient kernel.
    """
    z_shift, x_shift = fs_gradient_shift_maps(I, filterSize=filterSize)
    center_z = z_shift.shape[0] // 2
    center_x = z_shift.shape[1] // 2
    return localizeFSGradientFromMaps(z_shift, x_shift, center_z, center_x)
