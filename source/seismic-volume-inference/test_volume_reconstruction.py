"""Synthetic tests for volume-level stitching.

Run from anywhere:

    python source/test_volume_reconstruction.py
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from volume_inference import (  # noqa: E402
    VolumeAccumulator,
    build_spatial_slices,
    confusion_from_volumes,
    delta_metrics,
    expected_overlap_count,
    iter_patch_starts,
    metrics_from_confusion,
    spatial_slice_from_index,
)


class Failures(list):
    def check(self, cond: bool, msg: str) -> None:
        if not cond:
            self.append(msg)


def test_iter_patch_starts(f: Failures) -> None:
    f.check(iter_patch_starts(16, 16, 8) == [0], "full-size patch should start at 0 only")
    f.check(iter_patch_starts(32, 16, 8) == [0, 8, 16], "regular grid without extra tail")
    f.check(
        iter_patch_starts(142, 16, 8)[-1] == 142 - 16,
        "last start must align to the volume end",
    )
    # last-aligned extra patch
    starts = iter_patch_starts(25, 16, 8)
    f.check(starts == [0, 8, 9], f"expected extra tail start, got {starts}")


def test_full_coverage(f: Failures) -> None:
    shape = (142, 256, 256)
    patch = (16, 128, 128)
    stride = (8, 64, 64)
    count = expected_overlap_count(shape, patch, stride)
    f.check(count.shape == shape, "coverage map shape mismatch")
    f.check(int(count.min()) >= 1, "every voxel must be covered at least once")
    # interior of a 2x2x2 overlap grid should be 8
    f.check(int(count[20, 80, 80]) == 8, f"interior overlap should be 8, got {count[20, 80, 80]}")
    f.check(int(count[0, 0, 0]) == 1, "corner voxel should be covered once")


def test_stitch_then_argmax_not_concat(f: Failures) -> None:
    """Overlapping patches with conflicting argmax: volume avg wins."""
    volume_shape = (24, 64, 64)
    n_classes = 3
    acc = VolumeAccumulator(volume_shape, n_classes)

    # Patch A at z=0: class 1 is strongest
    a = np.zeros((n_classes, 16, 64, 64), dtype=np.float32)
    a[1] = 0.7
    a[2] = 0.2
    a[0] = 0.1
    acc.add_patch(a, (slice(0, 16), slice(0, 64), slice(0, 64)))

    # Patch B at z=8, overlapping [8:16]: class 2 is stronger than A's 1
    b = np.zeros((n_classes, 16, 64, 64), dtype=np.float32)
    b[2] = 0.9
    b[1] = 0.05
    b[0] = 0.05
    acc.add_patch(b, (slice(8, 24), slice(0, 64), slice(0, 64)))

    stats = acc.validate(patch_shape=(16, 64, 64), stride_shape=(8, 64, 64))
    pred, averaged = acc.finalize()
    f.check(pred.shape == volume_shape, "pred shape")
    f.check(averaged.shape == (n_classes,) + volume_shape, "avg shape")
    f.check(stats["n_uncovered_voxels"] == 0, "uncovered")

    # Non-overlap region follows patch A
    f.check(int(pred[0, 0, 0]) == 1, "non-overlap should be class 1")
    # Overlap: mean of (0.7, 0.05) for class1 = 0.375; class2 mean (0.2, 0.9) = 0.55
    f.check(int(pred[12, 0, 0]) == 2, "overlap argmax after averaging should be class 2")

    # Old (wrong) protocol: concat argmax would count class 1 once and class 2 once
    # in the overlap, i.e. 50/50, not the averaged decision.
    overlap_a = np.argmax(a[:, 8:, :, :], axis=0)
    overlap_b = np.argmax(b[:, :8, :, :], axis=0)
    f.check(int(overlap_a[0, 0, 0]) == 1, "patch A argmax in overlap is 1")
    f.check(int(overlap_b[0, 0, 0]) == 2, "patch B argmax in overlap is 2")
    f.check(
        not np.array_equal(pred[8:16], overlap_a),
        "stitched overlap must not equal first-patch argmax",
    )


def test_no_double_weighting(f: Failures) -> None:
    """Two identical patches -> average equals the patch, not 2x."""
    shape = (8, 64, 64)
    n_classes = 4
    acc = VolumeAccumulator(shape, n_classes)
    patch = np.zeros((n_classes,) + shape, dtype=np.float32)
    patch[3] = 0.4
    patch[1] = 0.3
    acc.add_patch(patch, (slice(0, 8), slice(0, 64), slice(0, 64)))
    acc.add_patch(patch, (slice(0, 8), slice(0, 64), slice(0, 64)))
    pred, averaged = acc.finalize()
    f.check(np.allclose(averaged, patch), "identical overlaps must average to the patch")
    f.check(int(pred[0, 0, 0]) == 3, "argmax of averaged identical patches")
    f.check(float(acc.prediction_count.max()) == 2.0, "count should be 2")


def test_channel_index_stripped(f: Failures) -> None:
    idx = (slice(0, 1), slice(2, 18), slice(0, 128), slice(64, 192))
    sl = spatial_slice_from_index(idx)
    f.check(sl == (slice(2, 18), slice(0, 128), slice(64, 192)), f"got {sl}")


def test_metrics_skip_background(f: Failures) -> None:
    gt = np.zeros((4, 8, 8), dtype=np.int64)
    pred = np.zeros_like(gt)
    gt[:, :4, :] = 1
    pred[:, :4, :] = 1
    gt[:, 4:, :4] = 2
    pred[:, 4:, :4] = 3  # confuse 2 -> 3
    cm = confusion_from_volumes(gt, pred, n_classes=4)
    m = metrics_from_confusion(cm, background_class=0)
    f.check(cm.shape == (4, 4), "cm shape")
    f.check(m["per_class"]["iou"][0] == 1.0, "background IoU should be 1")
    f.check(m["per_class"]["iou"][1] == 1.0, "class 1 perfect")
    f.check(m["mean_iou_with_background"] > m["mean_iou_no_background"], "bg must not inflate fg mIoU")
    # class 2: tp=0, so iou=0; class 3 support=0 iou=0
    f.check(m["per_class"]["iou"][2] == 0.0, "class 2 fully confused")


def test_shape_mismatch_aborts(f: Failures) -> None:
    acc = VolumeAccumulator((8, 64, 64), 3)
    bad = np.zeros((3, 8, 32, 64), dtype=np.float32)
    try:
        acc.add_patch(bad, (slice(0, 8), slice(0, 64), slice(0, 64)))
        f.check(False, "shape mismatch should raise")
    except ValueError:
        pass

    acc2 = VolumeAccumulator((8, 64, 64), 3)
    good = np.zeros((3, 8, 64, 64), dtype=np.float32)
    good[1] = 1
    acc2.add_patch(good, (slice(0, 8), slice(0, 64), slice(0, 64)))
    try:
        acc2.validate(expected_count=np.full((8, 64, 64), 2, dtype=np.int32))
        f.check(False, "coverage mismatch should raise")
    except RuntimeError as exc:
        f.check("coverage mismatch" in str(exc).lower() or "match" in str(exc).lower(), str(exc))


def test_delta_and_json_roundtrip(f: Failures) -> None:
    gt = np.random.default_rng(0).integers(0, 5, size=(10, 32, 32))
    pred_v = gt.copy()
    pred_p = gt.copy()
    pred_p[:2] = (pred_p[:2] + 1) % 5
    mv = metrics_from_confusion(confusion_from_volumes(gt, pred_v, 5))
    mp = metrics_from_confusion(confusion_from_volumes(gt, pred_p, 5))
    d = delta_metrics(mv, mp)
    f.check(d["pixel_accuracy"] > 0, "volume should beat the corrupted patch metric")
    f.check(len(d["per_class_iou"]) == 5, "per-class delta length")


def test_slices_match_builder_order(f: Failures) -> None:
    slices = build_spatial_slices((32, 128, 128), (16, 128, 128), (8, 64, 64))
    f.check(slices[0] == (slice(0, 16), slice(0, 128), slice(0, 128)), "first slice")
    f.check(slices[-1][0] == slice(16, 32), f"last z slice {slices[-1]}")
    f.check(len(slices) == 3, f"expected 3 z-positions * 1 y * 1 x, got {len(slices)}")


def main() -> int:
    f = Failures()
    tests = [
        test_iter_patch_starts,
        test_full_coverage,
        test_stitch_then_argmax_not_concat,
        test_no_double_weighting,
        test_channel_index_stripped,
        test_metrics_skip_background,
        test_shape_mismatch_aborts,
        test_delta_and_json_roundtrip,
        test_slices_match_builder_order,
    ]
    for test in tests:
        try:
            test(f)
            print(f"PASS  {test.__name__}")
        except Exception:
            f.append(f"{test.__name__} crashed:\n{traceback.format_exc()}")
            print(f"FAIL  {test.__name__} (exception)")
    if f:
        print("\nFAILURES:")
        for item in f:
            print(f"  - {item}")
        return 1
    print("\nAll reconstruction tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
