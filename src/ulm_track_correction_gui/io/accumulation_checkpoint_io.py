"""Efficient MAT reader and companion discovery for accumulation checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, whosmat

from ulm_track_correction_gui.core.accumulation_checkpoint import (
    SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS,
    AccumulationCheckpoint,
    SparseTrackContributions,
)

GLOBAL_MAP_NAMES = (
    "mapCounter",
    "mapCounter_AA",
    "VelocityDisplacement",
    "VelocityTrackMean",
)

FULL_VELOCITY_FIELDS = (
    "hardLocalVelocityNumerator",
    "hardTrackMeanVelocityNumerator",
)

GRID_FIELDS = (
    "inputShape",
    "mapShape",
    "inputPixelSizeZUm",
    "inputPixelSizeXUm",
    "ulmPixelSizeZUm",
    "ulmPixelSizeXUm",
    "ulmScaleZ",
    "ulmScaleX",
    "rasterRowColumnIndexBase",
    "sparseOffsetsIndexBase",
    "sparsePixelIndexConvention",
    "axisOrder",
)

CONTRIBUTION_FIELDS = (
    "trackIds",
    "hardOffsets",
    "hardRows",
    "hardCols",
    "aaOffsets",
    "aaRows",
    "aaCols",
    "aaDensityWeight",
    "hardLocalVelocityNumerator",
    "hardTrackMeanVelocityNumerator",
)

DISPLAY_DEFAULTS = {
    "counterPower": 1.0 / 3.0,
    "counterDisplayMax": 8.0,
    "counterAADisplayMax": 4.0,
    "velocityDisplayMax": 15.0,
}

LOADED_VARIABLES = (
    "checkpointSchemaVersion",
    "checkpointKind",
    "perTrackContributionMode",
    "accumulation_param",
    *GRID_FIELDS,
    *DISPLAY_DEFAULTS,
    *CONTRIBUTION_FIELDS,
)


@dataclass(frozen=True)
class CheckpointDiscovery:
    path: Path | None
    warnings: tuple[str, ...] = ()
    searched: tuple[Path, ...] = ()


def checkpoint_path_for_result(result_path: str | Path) -> Path:
    path = Path(result_path)
    stem = path.stem[:-8] if path.stem.endswith("_results") else path.stem
    return path.with_name(f"{stem}_accumulation_checkpoint.mat")


def _is_hdf5(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(8) == b"\x89HDF\r\n\x1a\n"
    except OSError:
        return False


def _mat_to_python(value: Any) -> Any:
    if hasattr(value, "_fieldnames"):
        return {name: _mat_to_python(getattr(value, name)) for name in value._fieldnames}
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.size == 1:
            return _mat_to_python(value.reshape(-1)[0])
        if value.dtype == object:
            return [_mat_to_python(item) for item in value.flat]
        if value.ndim == 0:
            return _mat_to_python(value.item())
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scalar_string(value: Any, *, field: str, default: str | None = None) -> str:
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"checkpoint field is missing: {field}")
    converted = _mat_to_python(value)
    if isinstance(converted, np.ndarray):
        if converted.dtype.kind in {"U", "S"}:
            text = "".join(str(item) for item in converted.reshape(-1))
        else:
            text = str(converted.reshape(-1)[0])
    else:
        text = str(converted)
    return text.strip()


def _scalar_int(value: Any, *, field: str) -> int:
    if value is None:
        raise ValueError(f"checkpoint field is missing: {field}")
    try:
        return int(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"checkpoint field {field} must be an integer") from exc


def _scalar_float(value: Any, *, field: str) -> float:
    if value is None:
        raise ValueError(f"checkpoint field is missing: {field}")
    try:
        output = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"checkpoint field {field} must be numeric") from exc
    if not np.isfinite(output):
        raise ValueError(f"checkpoint field {field} must be finite")
    return output


def _shape_vector(mat: dict[str, Any], field: str) -> tuple[int, int]:
    if field not in mat:
        raise ValueError(f"checkpoint field is missing: {field}")
    values = np.asarray(mat[field]).reshape(-1)
    if len(values) != 2:
        raise ValueError(f"checkpoint {field} must contain exactly two values")
    shape = tuple(int(value) for value in values)
    if min(shape) < 1:
        raise ValueError(f"checkpoint {field} must contain positive values")
    return shape


def _vector(
    mat: dict[str, Any],
    field: str,
    dtype: np.dtype,
    *,
    optional_empty: bool = False,
) -> np.ndarray:
    if field not in mat:
        if optional_empty:
            return np.empty(0, dtype=dtype)
        raise ValueError(f"checkpoint field is missing: {field}")
    return np.asarray(mat[field], dtype=dtype).reshape(-1)


def _load_selected_variables(path: Path) -> tuple[dict[str, Any], dict[str, tuple]]:
    if _is_hdf5(path):
        raise ValueError(
            f"{path.name} is a MATLAB v7.3 (HDF5) checkpoint, which is not "
            "supported yet. Re-save it in MATLAB with save(..., '-v7')."
        )
    try:
        inventory = {name: (shape, kind) for name, shape, kind in whosmat(path)}
        mat = loadmat(
            path,
            squeeze_me=False,
            struct_as_record=False,
            variable_names=LOADED_VARIABLES,
        )
    except NotImplementedError as exc:
        raise ValueError(
            f"{path.name} looks like a MATLAB v7.3 (HDF5) checkpoint. "
            "Re-save it in MATLAB with save(..., '-v7')."
        ) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not read accumulation checkpoint {path.name}: {exc}") from exc
    return mat, inventory


def _validate_grid_metadata(mat: dict[str, Any]) -> tuple[dict, tuple[int, int]]:
    input_shape = _shape_vector(mat, "inputShape")
    map_shape = _shape_vector(mat, "mapShape")
    pixel_sizes = {
        field: _scalar_float(mat.get(field), field=field)
        for field in (
            "inputPixelSizeZUm",
            "inputPixelSizeXUm",
            "ulmPixelSizeZUm",
            "ulmPixelSizeXUm",
        )
    }
    if min(pixel_sizes.values()) <= 0:
        raise ValueError("checkpoint pixel sizes must be positive")
    scale_z = _scalar_float(mat.get("ulmScaleZ"), field="ulmScaleZ")
    scale_x = _scalar_float(mat.get("ulmScaleX"), field="ulmScaleX")
    expected_z = pixel_sizes["inputPixelSizeZUm"] / pixel_sizes["ulmPixelSizeZUm"]
    expected_x = pixel_sizes["inputPixelSizeXUm"] / pixel_sizes["ulmPixelSizeXUm"]
    if not np.isclose(scale_z, expected_z, rtol=1e-8, atol=1e-12):
        raise ValueError("checkpoint ulmScaleZ does not match its pixel sizes")
    if not np.isclose(scale_x, expected_x, rtol=1e-8, atol=1e-12):
        raise ValueError("checkpoint ulmScaleX does not match its pixel sizes")
    expected_map_shape = (
        int(np.ceil((input_shape[0] - 1) * scale_z)) + 1,
        int(np.ceil((input_shape[1] - 1) * scale_x)) + 1,
    )
    if map_shape != expected_map_shape:
        raise ValueError(
            f"checkpoint mapShape {map_shape} does not match its input grid "
            f"and ULM scales (expected {expected_map_shape})"
        )

    if _scalar_int(mat.get("rasterRowColumnIndexBase"), field="rasterRowColumnIndexBase") != 0:
        raise ValueError("checkpoint rasterRowColumnIndexBase must be 0")
    if _scalar_int(mat.get("sparseOffsetsIndexBase"), field="sparseOffsetsIndexBase") != 0:
        raise ValueError("checkpoint sparseOffsetsIndexBase must be 0")
    if (
        _scalar_string(
            mat.get("sparsePixelIndexConvention"), field="sparsePixelIndexConvention"
        )
        != "row_col"
    ):
        raise ValueError("checkpoint sparsePixelIndexConvention must be row_col")
    if _scalar_string(mat.get("axisOrder"), field="axisOrder") != "z_x":
        raise ValueError("checkpoint axisOrder must be z_x")

    metadata = {
        "inputShape": input_shape,
        "mapShape": map_shape,
        **pixel_sizes,
        "ulmScaleZ": scale_z,
        "ulmScaleX": scale_x,
        "rasterRowColumnIndexBase": 0,
        "sparseOffsetsIndexBase": 0,
        "sparsePixelIndexConvention": "row_col",
        "axisOrder": "z_x",
    }
    return metadata, map_shape


def _validate_global_map_inventory(
    inventory: dict[str, tuple],
    map_shape: tuple[int, int],
) -> None:
    for field in GLOBAL_MAP_NAMES:
        if field not in inventory:
            raise ValueError(f"checkpoint global map is missing: {field}")
        stored_shape = tuple(int(value) for value in inventory[field][0])
        if stored_shape != map_shape:
            raise ValueError(
                f"checkpoint global map {field} has shape {stored_shape}, "
                f"expected mapShape {map_shape}"
            )


def _validate_sparse_inventory(
    inventory: dict[str, tuple],
    mode: str,
) -> None:
    expected_types = {
        "trackIds": "int64",
        "hardOffsets": "uint64",
        "hardRows": "uint32",
        "hardCols": "uint32",
        "aaOffsets": "uint64",
        "aaRows": "uint32",
        "aaCols": "uint32",
        "aaDensityWeight": "single",
    }
    if mode == "full":
        expected_types.update({field: "single" for field in FULL_VELOCITY_FIELDS})
    for field, expected in expected_types.items():
        if field not in inventory:
            raise ValueError(f"checkpoint field is missing: {field}")
        actual = inventory[field][1]
        if actual != expected:
            raise ValueError(
                f"checkpoint field {field} must use MATLAB {expected}, got {actual}"
            )


def _display_params(mat: dict[str, Any]) -> dict:
    nested = _mat_to_python(mat.get("accumulation_param"))
    params = dict(nested) if isinstance(nested, dict) else {}
    for field, default in DISPLAY_DEFAULTS.items():
        value = mat.get(field)
        params[field] = (
            _scalar_float(value, field=field)
            if value is not None
            else float(params.get(field, default))
        )
        if not np.isfinite(params[field]) or params[field] <= 0:
            raise ValueError(f"checkpoint display parameter {field} must be positive")
    return params


def load_accumulation_checkpoint(path: str | Path) -> AccumulationCheckpoint:
    """Load schema-v2 sparse contributions without loading global maps."""

    source = Path(path)
    mat, inventory = _load_selected_variables(source)
    version = _scalar_int(mat.get("checkpointSchemaVersion"), field="checkpointSchemaVersion")
    if version not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported checkpoint schema version {version}; "
            "supported version is 2"
        )
    kind = _scalar_string(mat.get("checkpointKind"), field="checkpointKind")
    if kind != "pala_accumulation":
        raise ValueError(f"unsupported checkpointKind: {kind}")
    mode = _scalar_string(
        mat.get("perTrackContributionMode"),
        field="perTrackContributionMode",
    ).lower()
    if mode not in {"none", "density_only", "full"}:
        raise ValueError(f"unsupported perTrackContributionMode: {mode}")

    metadata, map_shape = _validate_grid_metadata(mat)
    _validate_global_map_inventory(inventory, map_shape)
    for shape_field in ("inputShape", "mapShape"):
        if inventory.get(shape_field, (None, None))[1] != "uint64":
            raise ValueError(f"checkpoint field {shape_field} must use MATLAB uint64")
    params = _display_params(mat)

    contributions = None
    if mode != "none":
        _validate_sparse_inventory(inventory, mode)
        contributions = SparseTrackContributions(
            mode=mode,
            track_ids=_vector(mat, "trackIds", np.int64),
            hard_offsets=_vector(mat, "hardOffsets", np.uint64),
            hard_rows=_vector(mat, "hardRows", np.uint32, optional_empty=True),
            hard_cols=_vector(mat, "hardCols", np.uint32, optional_empty=True),
            aa_offsets=_vector(mat, "aaOffsets", np.uint64),
            aa_rows=_vector(mat, "aaRows", np.uint32, optional_empty=True),
            aa_cols=_vector(mat, "aaCols", np.uint32, optional_empty=True),
            aa_density_weight=_vector(mat, "aaDensityWeight", np.float32, optional_empty=True),
            hard_local_velocity_numerator=(
                _vector(mat, "hardLocalVelocityNumerator", np.float32)
                if mode == "full"
                else None
            ),
            hard_track_mean_velocity_numerator=(
                _vector(mat, "hardTrackMeanVelocityNumerator", np.float32)
                if mode == "full"
                else None
            ),
        ).validate(map_shape)

    return AccumulationCheckpoint(
        schema_version=version,
        mode=mode,
        map_shape=map_shape,
        input_shape=metadata["inputShape"],
        ulm_scale_z=metadata["ulmScaleZ"],
        ulm_scale_x=metadata["ulmScaleX"],
        params=params,
        metadata=metadata,
        contributions=contributions,
        source_path=source,
    )


def _result_checkpoint_reference(result_path: Path) -> str | None:
    if not result_path.exists() or _is_hdf5(result_path):
        return None
    try:
        mat = loadmat(
            result_path,
            squeeze_me=False,
            struct_as_record=False,
            variable_names=("accumulation_checkpoint_file",),
        )
    except (OSError, TypeError, ValueError, NotImplementedError):
        return None
    value = mat.get("accumulation_checkpoint_file")
    if value is None or np.asarray(value).size == 0:
        return None
    return _scalar_string(value, field="accumulation_checkpoint_file") or None


def _resolved_path(raw_path: str | Path, base_directory: Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    return path if path.is_absolute() else base_directory / path


def discover_accumulation_checkpoint(
    result_path: str | Path,
    metadata: dict | None = None,
) -> CheckpointDiscovery:
    """Discover a companion checkpoint without making it session-critical."""

    result = Path(result_path)
    metadata = dict(metadata or {})
    warnings: list[str] = []
    searched: list[Path] = []
    seen: set[Path] = set()

    def try_candidate(candidate: Path, *, explicit: bool = False) -> Path | None:
        candidate = candidate.resolve(strict=False)
        if candidate in seen:
            return None
        seen.add(candidate)
        searched.append(candidate)
        if candidate.is_file():
            return candidate
        if explicit:
            warnings.append(f"Referenced accumulation checkpoint was not found: {candidate}")
        return None

    reference = _result_checkpoint_reference(result)
    if reference:
        found = try_candidate(
            _resolved_path(reference, result.parent),
            explicit=True,
        )
        if found is not None:
            return CheckpointDiscovery(found, tuple(warnings), tuple(searched))

    if result.stem.endswith("_results"):
        found = try_candidate(checkpoint_path_for_result(result))
        if found is not None:
            return CheckpointDiscovery(found, tuple(warnings), tuple(searched))

    stored_checkpoint = metadata.get("accumulation_checkpoint_path")
    if stored_checkpoint:
        found = try_candidate(
            _resolved_path(stored_checkpoint, result.parent),
            explicit=True,
        )
        if found is not None:
            return CheckpointDiscovery(found, tuple(warnings), tuple(searched))

    source_value = metadata.get("source_path")
    if source_value:
        source_result = _resolved_path(source_value, result.parent)
        if source_result.resolve(strict=False) != result.resolve(strict=False):
            if not source_result.is_file():
                warnings.append(f"Original result MAT was not found: {source_result}")
            else:
                source_reference = _result_checkpoint_reference(source_result)
                if source_reference:
                    found = try_candidate(
                        _resolved_path(source_reference, source_result.parent),
                        explicit=True,
                    )
                    if found is not None:
                        return CheckpointDiscovery(found, tuple(warnings), tuple(searched))
                if source_result.stem.endswith("_results"):
                    found = try_candidate(checkpoint_path_for_result(source_result))
                    if found is not None:
                        return CheckpointDiscovery(found, tuple(warnings), tuple(searched))

    return CheckpointDiscovery(None, tuple(warnings), tuple(searched))


__all__ = [
    "CheckpointDiscovery",
    "FULL_VELOCITY_FIELDS",
    "GLOBAL_MAP_NAMES",
    "checkpoint_path_for_result",
    "discover_accumulation_checkpoint",
    "load_accumulation_checkpoint",
]
