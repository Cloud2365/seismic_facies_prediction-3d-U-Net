"""Volume-level patch stitching and segmentation metrics.

Numpy-only so reconstruction can be unit-tested without GPU / pytorch3dunet.

The previous test pipeline did:

    argmax(patch) -> concatenate overlapping patches -> metric

That double-counts interior voxels and is not a volume metric. The correct
protocol is:

    accumulate probabilities (or logits) at original coordinates
    -> average overlaps
    -> argmax on the reconstructed volume
    -> metric on the full test cube
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def iter_patch_starts(size: int, patch: int, stride: int) -> list[int]:
    """Same index generation as pytorch3dunet SliceBuilder._gen_indices."""
    if size < patch:
        raise ValueError(
            f"Volume size {size} is smaller than patch size {patch}"
        )
    starts = list(range(0, size - patch + 1, stride))
    last = starts[-1]
    if last + patch < size:
        starts.append(size - patch)
    return starts


def build_spatial_slices(
    volume_shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
    stride_shape: tuple[int, int, int],
) -> list[tuple[slice, slice, slice]]:
    """Build DxHxW slices covering `volume_shape` with the given patch/stride."""
    d, h, w = volume_shape
    pd, ph, pw = patch_shape
    sd, sh, sw = stride_shape
    slices: list[tuple[slice, slice, slice]] = []
    for z in iter_patch_starts(d, pd, sd):
        for y in iter_patch_starts(h, ph, sh):
            for x in iter_patch_starts(w, pw, sw):
                slices.append(
                    (
                        slice(z, z + pd),
                        slice(y, y + ph),
                        slice(x, x + pw),
                    )
                )
    return slices


def expected_overlap_count(
    volume_shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
    stride_shape: tuple[int, int, int],
) -> np.ndarray:
    """How many patches should cover each voxel."""
    count = np.zeros(volume_shape, dtype=np.int32)
    for sl in build_spatial_slices(volume_shape, patch_shape, stride_shape):
        count[sl] += 1
    return count


def spatial_slice_from_index(index) -> tuple[slice, slice, slice]:
    """Drop an optional channel slice from a dataset index."""
    if isinstance(index, (list, tuple)) and len(index) == 4:
        index = index[1:]
    if len(index) != 3:
        raise ValueError(
            f"Expected a 3D spatial index (D, H, W), got {index!r}"
        )
    return tuple(index)


class VolumeAccumulator:
    """Accumulate overlapping patch probabilities/logits into a full volume.

    Stores prediction_sum [C, D, H, W] and prediction_count [D, H, W].
    Argmax is applied only after averaging.
    """

    def __init__(self, volume_shape: tuple[int, int, int], n_classes: int):
        if len(volume_shape) != 3:
            raise ValueError(
                f"volume_shape must be (D, H, W), got {volume_shape}"
            )
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        self.volume_shape = tuple(int(x) for x in volume_shape)
        self.n_classes = int(n_classes)
        self.prediction_sum = np.zeros(
            (self.n_classes,) + self.volume_shape, dtype=np.float32
        )
        self.prediction_count = np.zeros(self.volume_shape, dtype=np.float32)
        self.n_patches = 0

    def add_patch(self, values: np.ndarray, spatial_slice) -> None:
        """Add one patch of shape [C, d, h, w] at `spatial_slice`."""
        spatial_slice = spatial_slice_from_index(spatial_slice)
        values = np.asarray(values)
        if values.ndim != 4:
            raise ValueError(
                "Patch values must be [C, D, H, W], "
                f"got shape {values.shape}"
            )
        if values.shape[0] != self.n_classes:
            raise ValueError(
                f"Patch has {values.shape[0]} channels, "
                f"expected {self.n_classes}"
            )
        dest = self.prediction_sum[(slice(None),) + spatial_slice]
        if dest.shape[1:] != values.shape[1:]:
            raise ValueError(
                "Patch spatial shape does not match destination slice: "
                f"patch={values.shape[1:]}, dest={dest.shape[1:]}, "
                f"slice={spatial_slice}"
            )
        dest += values.astype(np.float32, copy=False)
        self.prediction_count[spatial_slice] += 1.0
        self.n_patches += 1

    def coverage_stats(self) -> dict[str, Any]:
        count = self.prediction_count
        return {
            "volume_shape": list(self.volume_shape),
            "n_patches": int(self.n_patches),
            "n_classes": int(self.n_classes),
            "prediction_count_min": float(count.min()) if count.size else 0.0,
            "prediction_count_max": float(count.max()) if count.size else 0.0,
            "prediction_count_mean": float(count.mean()) if count.size else 0.0,
            "n_uncovered_voxels": int(np.count_nonzero(count == 0)),
        }

    def validate(
        self,
        expected_count: np.ndarray | None = None,
        patch_shape: tuple[int, int, int] | None = None,
        stride_shape: tuple[int, int, int] | None = None,
    ) -> dict[str, Any]:
        """Abort with a clear message if reconstruction is inconsistent."""
        stats = self.coverage_stats()
        if self.prediction_sum.shape[1:] != self.volume_shape:
            raise RuntimeError(
                "Reconstructed prediction_sum spatial shape "
                f"{self.prediction_sum.shape[1:]} != volume {self.volume_shape}"
            )
        if self.prediction_count.shape != self.volume_shape:
            raise RuntimeError(
                "prediction_count shape "
                f"{self.prediction_count.shape} != volume {self.volume_shape}"
            )

        if expected_count is None and patch_shape is not None and stride_shape is not None:
            expected_count = expected_overlap_count(
                self.volume_shape, patch_shape, stride_shape
            )

        if expected_count is not None:
            expected_count = np.asarray(expected_count)
            if expected_count.shape != self.volume_shape:
                raise RuntimeError(
                    "expected_count shape "
                    f"{expected_count.shape} != volume {self.volume_shape}"
                )
            if not np.array_equal(
                self.prediction_count.astype(np.int64),
                expected_count.astype(np.int64),
            ):
                diff = np.abs(
                    self.prediction_count.astype(np.int64)
                    - expected_count.astype(np.int64)
                )
                raise RuntimeError(
                    "Reconstruction coverage mismatch: prediction_count does "
                    "not match the expected overlapping-patch map. "
                    f"max |diff|={int(diff.max())}, "
                    f"n_mismatch={int(np.count_nonzero(diff))}, "
                    f"observed min/max/mean="
                    f"{stats['prediction_count_min']:.3f}/"
                    f"{stats['prediction_count_max']:.3f}/"
                    f"{stats['prediction_count_mean']:.3f}."
                )
            stats["expected_count_min"] = int(expected_count.min())
            stats["expected_count_max"] = int(expected_count.max())
            stats["expected_count_mean"] = float(expected_count.mean())

        uncovered = stats["n_uncovered_voxels"]
        if uncovered:
            raise RuntimeError(
                f"Reconstruction failed: {uncovered} voxels inside the test "
                "volume were never covered by a patch "
                f"(volume_shape={self.volume_shape}, "
                f"n_patches={self.n_patches}). "
                "Check patch_shape / stride_shape / axis order."
            )
        if self.n_patches == 0:
            raise RuntimeError("Reconstruction failed: no patches were added.")
        return stats

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        """Average overlaps, then argmax. Returns (pred[D,H,W], avg[C,D,H,W])."""
        count = self.prediction_count
        if np.any(count == 0):
            raise RuntimeError(
                "Cannot finalize reconstruction: prediction_count contains "
                f"{int(np.count_nonzero(count == 0))} zeros. "
                "Call validate() for a detailed error."
            )
        averaged = self.prediction_sum / count[None, ...]
        pred = np.argmax(averaged, axis=0).astype(np.int64)
        if pred.shape != self.volume_shape:
            raise RuntimeError(
                f"Argmax volume shape {pred.shape} != {self.volume_shape}"
            )
        return pred, averaged


def confusion_from_volumes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Confusion matrix of a full volume. Rows = true, columns = predicted."""
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64, copy=False)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.int64, copy=False)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred size mismatch: {y_true.shape} vs {y_pred.shape}"
        )
    mask = (y_true >= 0) & (y_true < n_classes) & (y_pred >= 0) & (y_pred < n_classes)
    idx = n_classes * y_true[mask] + y_pred[mask]
    cm = np.bincount(idx, minlength=n_classes * n_classes)
    return cm.reshape(n_classes, n_classes).astype(np.int64)


def metrics_from_confusion(
    cm: np.ndarray,
    background_class: int = 0,
) -> dict[str, Any]:
    """Pixel / class accuracy, IoU, precision, recall, F1 from a confusion matrix."""
    cm = np.asarray(cm, dtype=np.float64)
    n_classes = cm.shape[0]
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    fp = pred_count - tp
    fn = support - tp
    union = tp + fp + fn

    precision = np.divide(
        tp, tp + fp, out=np.full(n_classes, np.nan), where=(tp + fp) > 0
    )
    recall = np.divide(
        tp, tp + fn, out=np.full(n_classes, np.nan), where=(tp + fn) > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.full(n_classes, np.nan),
        where=(precision + recall) > 0,
    )
    iou = np.divide(tp, union, out=np.full(n_classes, np.nan), where=union > 0)
    class_accuracy = np.divide(
        tp, support, out=np.full(n_classes, np.nan), where=support > 0
    )

    pixel_accuracy = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0

    fg = np.array([c != background_class for c in range(n_classes)])

    def _finite(values, mask=None):
        arr = np.asarray(values, dtype=np.float64)
        if mask is not None:
            arr = arr[mask]
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(arr.mean())

    def _zero_fill(values):
        arr = np.asarray(values, dtype=np.float64)
        return np.where(np.isfinite(arr), arr, 0.0).tolist()

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    return {
        "n_classes": int(n_classes),
        "background_class": int(background_class),
        "pixel_accuracy": pixel_accuracy,
        "mean_class_accuracy": _finite(class_accuracy),
        "mean_class_accuracy_no_background": _finite(class_accuracy, fg),
        "mean_iou_with_background": _finite(iou),
        "mean_iou_no_background": _finite(iou, fg),
        "mean_iou_present_no_background": _finite(iou, fg),
        "mean_precision_no_background": _finite(precision, fg),
        "mean_recall_no_background": _finite(recall, fg),
        "mean_f1_no_background": _finite(f1, fg),
        "per_class": {
            "accuracy": _zero_fill(class_accuracy),
            "iou": _zero_fill(iou),
            "precision": _zero_fill(precision),
            "recall": _zero_fill(recall),
            "f1": _zero_fill(f1),
            "support": support.astype(np.int64).tolist(),
        },
        "confusion_matrix": cm.astype(np.int64).tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    }


def delta_metrics(volume_metrics: dict, patch_metrics: dict) -> dict[str, Any]:
    """volume - patch for the headline numbers and per-class IoU."""
    keys = [
        "pixel_accuracy",
        "mean_class_accuracy",
        "mean_iou_no_background",
        "mean_iou_with_background",
    ]
    out = {}
    for key in keys:
        if key in volume_metrics and key in patch_metrics:
            out[key] = float(volume_metrics[key] - patch_metrics[key])
    v_iou = volume_metrics.get("per_class", {}).get("iou", [])
    p_iou = patch_metrics.get("per_class", {}).get("iou", [])
    n = min(len(v_iou), len(p_iou))
    out["per_class_iou"] = [float(v_iou[i] - p_iou[i]) for i in range(n)]
    return out


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def dump_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")
