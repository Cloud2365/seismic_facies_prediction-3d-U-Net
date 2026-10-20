import numpy as np

from volume_inference import (
    VolumeAccumulator,
    build_spatial_slices,
    confusion_from_volumes,
    expected_overlap_count,
    metrics_from_confusion,
)


def test_stitching_recovers_volume_and_coverage():
    shape, patch, stride = (10, 20, 24), (4, 8, 8), (2, 4, 4)
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 3, size=shape)
    acc = VolumeAccumulator(shape, 3)
    for sl in build_spatial_slices(shape, patch, stride):
        onehot = np.eye(3, dtype=np.float32)[gt[sl]].transpose(3, 0, 1, 2)
        acc.add_patch(onehot, sl)
    acc.validate(expected_count=expected_overlap_count(shape, patch, stride),
                 patch_shape=patch, stride_shape=stride)
    pred, _ = acc.finalize()
    assert (pred == gt).all()
    m = metrics_from_confusion(confusion_from_volumes(gt, pred, 3))
    assert m["pixel_accuracy"] == 1.0
