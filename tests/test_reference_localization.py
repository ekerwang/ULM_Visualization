import numpy as np

from ulm_track_correction_gui.reference_processing.localization_funcs.ULM_localization2D_NCCbased import (  # noqa: E501
    ULM_localization2D_NCCbased,
)
from ulm_track_correction_gui.reference_processing.wrappers import (
    RADIAL_NCC_METHOD,
    localization_wrapper,
)


def asymmetric_gaussian(
    z0: float,
    x0: float,
    *,
    fwhm_z: float = 5.0,
    fwhm_x: float = 7.0,
    nz: int = 41,
    nx: int = 47,
) -> np.ndarray:
    z_grid, x_grid = np.meshgrid(
        np.arange(1, nz + 1, dtype=float),
        np.arange(1, nx + 1, dtype=float),
        indexing="ij",
    )
    return np.exp(
        -4.0
        * np.log(2.0)
        * (
            ((z_grid - z0) ** 2) / fwhm_z**2
            + ((x_grid - x0) ** 2) / fwhm_x**2
        )
    )


def radial_ncc_params() -> dict:
    return {
        "method": RADIAL_NCC_METHOD,
        "psfSizeZAxis": 5.0,
        "psfSizeXAxis": 7.0,
        "psfWindowSize": 9,
        "ampThreshold": 0.05,
        "corrThreshold": 0.4,
    }


def test_radial_ncc_refines_asymmetric_complex_gaussian() -> None:
    z0, x0 = 17.35, 28.7
    frame = asymmetric_gaussian(z0, x0) * (1.0 + 0.25j)

    result = ULM_localization2D_NCCbased(
        frame,
        PSFSize=[5.0, 7.0],
        PSFWindowSize=9,
        AmpThreshold=0.05,
        CorrThreshold=0.4,
        LocMethod="radial",
    )

    assert result.shape == (1, 4)
    assert result.dtype == np.float64
    assert abs(result[0, 1] - z0) < 0.25
    assert abs(result[0, 2] - x0) < 0.25
    assert result[0, 3] == 1.0


def test_radial_ncc_wrapper_preserves_frame_base_for_stack() -> None:
    frame = asymmetric_gaussian(17.35, 28.7)
    stack = np.stack([frame, 0.8 * frame], axis=2)

    localized = localization_wrapper(stack, radial_ncc_params())

    assert set(localized) == {0, 1}
    assert localized[0].shape == (1, 4)
    assert localized[1].shape == (1, 4)
    assert localized[0][0, 3] == 1.0
    assert localized[1][0, 3] == 2.0


def test_ncc_default_remains_fs_gradient() -> None:
    frame = asymmetric_gaussian(17.35, 28.7)
    kwargs = {
        "PSFSize": [5.0, 7.0],
        "PSFWindowSize": 9,
        "AmpThreshold": 0.05,
        "CorrThreshold": 0.4,
        "FSFilterSize": 7,
    }

    default_result = ULM_localization2D_NCCbased(frame, **kwargs)
    explicit_result = ULM_localization2D_NCCbased(
        frame,
        **kwargs,
        LocMethod="FS_Gradient",
    )

    np.testing.assert_array_equal(default_result, explicit_result)


def test_radial_ncc_rejects_peak_without_complete_refinement_roi() -> None:
    edge_frame = asymmetric_gaussian(2.2, 28.7)

    result = ULM_localization2D_NCCbased(
        edge_frame,
        PSFSize=[5.0, 7.0],
        PSFWindowSize=9,
        AmpThreshold=0.05,
        CorrThreshold=0.4,
        LocMethod="radial",
    )

    assert result.shape == (0, 4)
