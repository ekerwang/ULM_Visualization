"""NCC-based 2D microbubble localization (lab-pipeline framework).


Python port of the localization stage of the MATLAB custom tracker
(ULM_2D_customTracker): PSF_Gaussian.m + AttentionMap_NCC_Correlation.m +
MB_Localization_FS_Gradient.m. The NCC detection stage can be paired with
either FS-gradient or radial-symmetry sub-pixel localization.


Unlike ULM_localization2D_NonNCC (which sorts intensity maxima and keeps the
numberOfParticles brightest), this framework:

1. Builds a Gaussian PSF template from PSFSize (FWHM in pixels) and
   PSFWindowSize.
2. Computes the attention map: normalized cross-correlation between the
   magnitude frame and the PSF, cropped to the original frame size
   (MATLAB ``normxcorr2`` + center crop; values in [-1, 1]).
3. Zeroes the attention map below CorrThreshold, then takes its regional
   maxima (``imregionalmax`` equivalent) as bubble candidates.
4. Refines each candidate to sub-pixel precision with either Farid-Simoncelli
   derivative filters or radial symmetry.
5. Discards bubbles whose amplitude is below AmpThreshold * max(frame).


Example
-------
>>> MatTracking = ULM_localization2D_NCCbased(MatIn, PSFSize=[4, 3.5],
...                                           PSFWindowSize=5,
...                                           AmpThreshold=0.05,
...                                           CorrThreshold=0.54)
"""


import numpy as np
from skimage.feature import match_template
from skimage.morphology import local_maxima

from .localizeFSGradient import fs_gradient_shift_maps
from .localizeRadialSymmetry import localizeRadialSymmetry


_FS_GRADIENT_ALIASES = {
    "fs_gradient",
    "fsgradient",
    "fs gradient",
    "fs_gradient_ncc",
}
_RADIAL_ALIASES = {"radial", "radial_symmetry", "radial symmetry", "radial_ncc"}


def _normalize_loc_method(method: str) -> str:
    key = str(method or "").strip().lower()
    if key in _FS_GRADIENT_ALIASES:
        return "fs_gradient"
    if key in _RADIAL_ALIASES:
        return "radial"
    raise ValueError(
        "Wrong LocMethod selected. Supported NCC refinement methods are "
        "'FS_Gradient' and 'radial'."
    )


def _matlab_round(value: float) -> int:
    """Round half away from zero, matching MATLAB's ``round``."""

    return int(np.sign(value) * np.floor(abs(float(value)) + 0.5))


def _normalize_pair(values, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 1:
        values = np.repeat(values, 2)
    if values.size != 2 or np.any(values <= 0):
        raise ValueError(f"{name} must be one or two positive values")
    return values


def psf_gaussian(fwhm, window_size=None):
    """Build a unit-sum 2D Gaussian PSF (port of MB_PSF/PSF_Gaussian.m).

    Parameters
    ----------
    fwhm : float or (2,) sequence
        PSF full-width-half-max in pixels, [axial(z), lateral(x)]. A scalar is
        used for both axes.
    window_size : int, (2,) sequence or None
        PSF kernel size in pixels, forced odd. None => 2*ceil(3*sigma)+1
        (3-sigma window), like the MATLAB default.

    Returns
    -------
    psf : ndarray, shape (window_z, window_x)
        Gaussian normalized so that psf.sum() == 1.
    """
    fwhm = _normalize_pair(fwhm, "PSF size (FWHM)")

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    if window_size is None:
        window = (2 * np.ceil(3.0 * sigma) + 1).astype(int)
    else:
        window = np.asarray(window_size, dtype=float).ravel()
        if window.size == 1:
            window = np.repeat(window, 2)
        if window.size != 2 or np.any(window < 1):
            raise ValueError("PSF window size must be one or two values >= 1")
        window = window.astype(int)
    window = (2 * np.ceil((window - 1) / 2) + 1).astype(int)  # force odd

    center = np.ceil(window / 2.0)  # 1-based center, like MATLAB
    zz, xx = np.meshgrid(np.arange(1, window[0] + 1),
                         np.arange(1, window[1] + 1), indexing='ij')
    ratio = (((zz - center[0]) ** 2) / (2.0 * sigma[0] ** 2)
             + ((xx - center[1]) ** 2) / (2.0 * sigma[1] ** 2))
    psf = np.exp(-ratio)
    return psf / psf.sum()


def ncc_attention_map(frame, psf):
    """Same-size NCC map (MATLAB normxcorr2 + center crop equivalent).

    Index (z, x) of the returned map corresponds to the PSF centered on pixel
    (z, x) of ``frame``; values are correlation coefficients in [-1, 1].
    """
    if frame.shape[0] < psf.shape[0] or frame.shape[1] < psf.shape[1]:
        raise ValueError(
            f"frame {frame.shape} is smaller than the PSF window {psf.shape}")
    return match_template(frame, psf, pad_input=True,
                          mode='constant', constant_values=0)


def ULM_localization2D_NCCbased(MatIn, *, PSFSize, PSFWindowSize=None,
                                AmpThreshold=0.05, CorrThreshold=0.54,
                                FSFilterSize=7, LocMethod="FS_Gradient"):
    """Detect and localize microbubbles via PSF normalized cross-correlation.


    Parameters
    ----------
    MatIn : array_like, shape (height, width) or (height, width, numberOfFrames)
        The sequence containing all the images. Complex data is converted to
        magnitude.
    PSFSize : float or (2,) sequence  (required, keyword-only)
        PSF FWHM in pixels, [axial(z), lateral(x)]. Lab pipeline default:
        [4, 3.5].
    PSFWindowSize : int, (2,) sequence or None, default None
        PSF kernel size in pixels (forced odd). None => 3-sigma auto window.
        Lab pipeline default: 5.
    AmpThreshold : float in [0, 1], default 0.05
        Normalized amplitude threshold. Candidates whose magnitude is below
        AmpThreshold * max(frame magnitude) are discarded.
    CorrThreshold : float, default 0.54
        Attention-map threshold (absolute NCC value). The correlation map is
        zeroed below this value before peak detection, so only peaks with
        NCC >= CorrThreshold become bubbles.
    FSFilterSize : {5, 7}, default 7
        Farid-Simoncelli derivative filter tap count.
    LocMethod : {'FS_Gradient', 'radial'}, default 'FS_Gradient'
        Sub-pixel localization method applied to peaks from the NCC attention
        map. ``'radial'`` uses a PSF-size ROI around each NCC peak.


    Returns
    -------
    MatTracking : ndarray, shape (N, 4)
        Same layout as ULM_localization2D_NonNCC, one row per bubble:
            column 0 : intensity (magnitude at the integer peak)
            column 1 : super-resolved axial coordinate (z), in pixels
            column 2 : super-resolved lateral coordinate (x), in pixels
            column 3 : frame number
        Coordinates and frame numbers are 1-based (MATLAB convention).
    """
    MatIn = np.asarray(MatIn)
    orig_dtype = MatIn.dtype
    out_dtype = orig_dtype if np.issubdtype(orig_dtype, np.floating) else np.float64

    amp_threshold = float(AmpThreshold)
    corr_threshold = float(CorrThreshold)
    if not 0.0 <= amp_threshold <= 1.0:
        raise ValueError("AmpThreshold must be in [0, 1]")

    loc_method = _normalize_loc_method(LocMethod)
    psf_size = _normalize_pair(PSFSize, "PSF size (FWHM)")
    fwhm_z = float(psf_size[0])
    fwhm_x = float(psf_size[1])
    psf = psf_gaussian(psf_size, PSFWindowSize)

    MatIn = np.abs(MatIn).astype(np.float64, copy=False)
    if MatIn.ndim == 2:
        MatIn = MatIn[:, :, np.newaxis]
    height, width, numberOfFrames = MatIn.shape

    # The NCC value is unreliable where the PSF window sticks out of the frame
    # (zero padding); peaks in that border ring are dropped. The MATLAB
    # pipeline avoids this by cropping the image upstream.
    border_z = psf.shape[0] // 2
    border_x = psf.shape[1] // 2
    if loc_method == "radial":
        half_z = _matlab_round(fwhm_z / 2.0)
        half_x = _matlab_round(fwhm_x / 2.0)
        radial_offsets_z = np.arange(-half_z, half_z + 1)
        radial_offsets_x = np.arange(-half_x, half_x + 1)
        border_z = max(border_z, half_z)
        border_x = max(border_x, half_x)
    interior = np.zeros((height, width), dtype=bool)
    interior[border_z:height - border_z, border_x:width - border_x] = True

    rows = []
    for f in range(numberOfFrames):
        frame = MatIn[:, :, f]
        frame[~np.isfinite(frame)] = 0.0

        # Attention map, thresholded before peak detection (like the MATLAB
        # MB_Localization_FS_Gradient.execute).
        corr_map = ncc_attention_map(frame, psf)
        corr_map[corr_map < corr_threshold] = 0.0

        peaks = local_maxima(corr_map, connectivity=2, allow_borders=True)
        # Guard against zero plateaus being flagged as regional maxima when
        # nothing survives the threshold.
        peaks &= corr_map >= max(corr_threshold, np.finfo(np.float64).tiny)
        peaks &= interior
        if not peaks.any():
            continue

        peak_z, peak_x = np.nonzero(peaks)
        amplitudes = frame[peak_z, peak_x]
        keep = amplitudes >= amp_threshold * frame.max()
        n_keep = int(np.count_nonzero(keep))
        if n_keep == 0:
            continue

        peak_z = peak_z[keep]
        peak_x = peak_x[keep]
        amplitudes = amplitudes[keep]

        if loc_method == "fs_gradient":
            z_shift, x_shift = fs_gradient_shift_maps(
                frame,
                filterSize=FSFilterSize,
            )
            shifts_z = z_shift[peak_z, peak_x]
            shifts_x = x_shift[peak_z, peak_x]
            shifts_z[~np.isfinite(shifts_z)] = 0.0
            shifts_x[~np.isfinite(shifts_x)] = 0.0

            out = np.empty((n_keep, 4), dtype=np.float64)
            out[:, 0] = amplitudes
            # +1 converts the 0-based numpy index to MATLAB's 1-based coordinate.
            out[:, 1] = peak_z + shifts_z + 1.0
            out[:, 2] = peak_x + shifts_x + 1.0
            out[:, 3] = f + 1
            rows.append(out)
            continue

        frame_rows = []
        for z_idx, x_idx, amplitude in zip(peak_z, peak_x, amplitudes):
            roi = frame[
                np.ix_(z_idx + radial_offsets_z, x_idx + radial_offsets_x)
            ]
            z_center, x_center = localizeRadialSymmetry(roi, fwhm_z, fwhm_x)
            if not np.isfinite(z_center) or not np.isfinite(x_center):
                continue
            if abs(z_center) > fwhm_z / 2.0 or abs(x_center) > fwhm_x / 2.0:
                continue
            frame_rows.append(
                [
                    amplitude,
                    z_idx + z_center + 1.0,
                    x_idx + x_center + 1.0,
                    f + 1,
                ]
            )

        if frame_rows:
            rows.append(np.asarray(frame_rows, dtype=np.float64))

    if rows:
        MatTracking = np.vstack(rows)
    else:
        MatTracking = np.zeros((0, 4), dtype=np.float64)

    return MatTracking.astype(out_dtype)
