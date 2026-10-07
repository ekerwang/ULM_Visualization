"""Radial-symmetry sub-pixel localization (Python port of localizeRadialSymmetry.m).

Performs localization using radial symmetry properties.

Created by Baptiste Heiles on 05/09/18, inspired from Raghuveer Parthasarathy,
The University of Oregon.

DATE 2020.07.22 - VERSION 1.1
AUTHORS: Arthur Chavignon, Baptiste Heiles, Vincent Hingot. CNRS, Sorbonne
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

Calculates the center of a 2D intensity distribution. Method: the gradient of a
function that has perfect radial symmetry will point towards the origin. We take
the local gradient and construct lines through any point with orientation
parallel to the local gradient. The origin is the point that minimizes the
distance between itself and all such lines.
"""

import numpy as np
from scipy.signal import convolve2d


def localizeRadialSymmetry(I, fwhmz=None, fwhmx=None):
    """Compute the center of radial symmetry of a 2D intensity distribution.

    Parameters
    ----------
    I : array_like, shape (Nz, Nx)
        2D intensity distribution. Size need not be an odd number of pixels
        along each dimension.
    fwhmz, fwhmx : float, optional
        Full width at half maximum in direction z and x (unused, kept for
        signature compatibility with the MATLAB version).

    Returns
    -------
    zc, xc : float
        The center of radial symmetry, in pixels, relative to the image center.
    """
    I = np.asarray(I, dtype=np.float64)
    Nz, Nx = I.shape

    # Grid midpoint coordinates: -n+0.5 : n-0.5 (length Nz-1 and Nx-1).
    # z increases "downward".
    zm_1d = np.arange(Nz - 1) - (Nz - 2) / 2.0
    xm_1d = np.arange(Nx - 1) - (Nx - 2) / 2.0
    zm, xm = np.meshgrid(zm_1d, xm_1d, indexing='ij')  # both (Nz-1, Nx-1)

    # Derivatives along 45-degree shifted coordinates (u and v).
    dIdu = I[:Nz - 1, 1:Nx] - I[1:Nz, :Nx - 1]   # gradient along u
    dIdv = I[:Nz - 1, :Nx - 1] - I[1:Nz, 1:Nx]   # gradient along v

    # Smooth the gradient of the I window with a simple 3x3 averaging filter
    # (conv2 'same' uses zero padding at the borders).
    h = np.ones((3, 3)) / 9.0
    fdu = convolve2d(dIdu, h, mode='same')
    fdv = convolve2d(dIdv, h, mode='same')
    dImag2 = fdu * fdu + fdv * fdv  # squared gradient magnitude

    # Slope of the gradient.
    with np.errstate(divide='ignore', invalid='ignore'):
        m = -(fdv + fdu) / (fdu - fdv)

    # If m is NaN (can happen when fdu == fdv) replace with un-smoothed gradient.
    nan_mask = np.isnan(m)
    if nan_mask.any():
        with np.errstate(divide='ignore', invalid='ignore'):
            unsmoothm = (dIdv + dIdu) / (dIdu - dIdv)
        m[nan_mask] = unsmoothm[nan_mask]

    # If it is still NaN, replace with zero (dealt with later).
    nan_mask = np.isnan(m)
    if nan_mask.any():
        m[nan_mask] = 0.0

    # Replace inf (can happen when fdu == fdv) by 10x the max finite slope.
    inf_mask = np.isinf(m)
    if inf_mask.any():
        finite_vals = m[~inf_mask]
        if finite_vals.size > 0:
            m[inf_mask] = 10.0 * np.max(finite_vals)
        else:
            # Fallback: replace m with the unsmoothed gradient.
            with np.errstate(divide='ignore', invalid='ignore'):
                m = (dIdv + dIdu) / (dIdu - dIdv)

    # z intercept of the line of slope m through each grid midpoint.
    b = zm - m * xm

    # Weight the intensity by squared gradient magnitude and inverse distance to
    # the gradient intensity centroid (initial guess of the center).
    sdI2 = np.sum(dImag2)
    zcentroid = np.sum(dImag2 * zm) / sdI2
    xcentroid = np.sum(dImag2 * xm) / sdI2
    with np.errstate(divide='ignore', invalid='ignore'):
        w = dImag2 / np.sqrt((zm - zcentroid) ** 2 + (xm - xcentroid) ** 2)

    # Least-squares minimization to find the translated coordinate-system origin.
    zc, xc = _lsradialcenterfit(m, b, w)
    return zc, xc


def _lsradialcenterfit(m, b, w):
    """Least-squares solution to determine the radial symmetry center.

    Inputs m, b, w are defined on a grid; w are the per-point weights. Determines
    the origin (xc, zc) such that lines z = m*x + b have minimal total distance^2
    to the origin.
    """
    wm2p1 = w / (m * m + 1)
    sw = np.sum(wm2p1)
    smmw = np.sum(m * m * wm2p1)
    smw = np.sum(m * wm2p1)
    smbw = np.sum(m * b * wm2p1)
    sbw = np.sum(b * wm2p1)
    det = smw * smw - smmw * sw
    xc = (smbw * sw - smw * sbw) / det   # relative to image center
    zc = (smbw * smw - smmw * sbw) / det  # relative to image center
    return zc, xc
