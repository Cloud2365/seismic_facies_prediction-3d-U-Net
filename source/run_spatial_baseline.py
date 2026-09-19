import argparse
import json
import os
import sys

import h5py
import numpy as np

# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from source.volume_inference import (
    confusion_from_volumes,
    metrics_from_confusion,
)


# ------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Spatial depth-only baseline for seismic facies"
    )

    parser.add_argument(
        "--train_h5",
        type=str,
        default="./temp_splits/spatial/train.h5",
        help="Train HDF5 file"
    )

    parser.add_argument(
        "--test_h5",
        type=str,
        default="./temp_splits/spatial/test.h5",
        help="Test HDF5 file"
    )

    parser.add_argument(
        "--n_classes",
        type=int,
        default=10,
        help="Number of classes"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./baseline_diagnostics",
        help="Directory for baseline results"
    )

    return parser.parse_args()


# ------------------------------------------------------------
# BUILD DEPTH PROFILE
# ------------------------------------------------------------

def build_depth_profile(
    train_labels: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    For each depth d, select the most frequent class
    over all train crosslines and lateral positions.
    """

    if train_labels.ndim != 3:
        raise ValueError(
            f"Expected labels with shape [crossline, depth, inline], "
            f"got {train_labels.shape}"
        )

    n_depth = train_labels.shape[1]

    depth_profile = np.empty(
        n_depth,
        dtype=np.int64
    )

    for depth in range(n_depth):

        labels_at_depth = train_labels[:, depth, :].reshape(-1)

        counts = np.bincount(
            labels_at_depth,
            minlength=n_classes
        )

        depth_profile[depth] = np.argmax(counts)

    return depth_profile


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    args = parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD TRAIN LABELS
    # --------------------------------------------------------

    print("=" * 80)
    print("SPATIAL BASELINE B0")
    print("=" * 80)

    print()
    print("Train H5:")
    print(args.train_h5)

    print()
    print("Test H5:")
    print(args.test_h5)

    with h5py.File(args.train_h5, "r") as f:

        if "label" not in f:
            raise RuntimeError(
                "Dataset 'label' not found in train HDF5."
            )

        train_labels = np.asarray(
            f["label"][:]
        )

    print()
    print(
        f"Train label shape: {train_labels.shape}"
    )

    # --------------------------------------------------------
    # BUILD PROFILE FROM TRAIN ONLY
    # --------------------------------------------------------

    depth_profile = build_depth_profile(
        train_labels,
        args.n_classes
    )

    print()
    print(
        f"Depth profile length: {len(depth_profile)}"
    )

    # --------------------------------------------------------
    # LOAD TEST LABELS
    # --------------------------------------------------------

    with h5py.File(args.test_h5, "r") as f:

        if "label" not in f:
            raise RuntimeError(
                "Dataset 'label' not found in test HDF5."
            )

        test_labels = np.asarray(
            f["label"][:]
        )

    print(
        f"Test label shape: {test_labels.shape}"
    )

    # --------------------------------------------------------
    # CHECK DEPTH COMPATIBILITY
    # --------------------------------------------------------

    if test_labels.shape[1] != len(depth_profile):
        raise ValueError(
            "Train and test have different depth dimensions: "
            f"{len(depth_profile)} vs {test_labels.shape[1]}"
        )

    # --------------------------------------------------------
    # PREDICT USING DEPTH ONLY
    # --------------------------------------------------------

    prediction = np.broadcast_to(
        depth_profile[None, :, None],
        test_labels.shape
    )

    prediction = np.asarray(
        prediction,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    cm = confusion_from_volumes(
        test_labels,
        prediction,
        args.n_classes
    )

    metrics = metrics_from_confusion(
        cm
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        f"Pixel Accuracy           : "
        f"{metrics['pixel_accuracy']:.6f}"
    )

    print(
        f"Mean IoU without class 0 : "
        f"{metrics['mean_iou_no_background']:.6f}"
    )

    print()
    print("Per-class IoU:")

    for class_id, iou in enumerate(
        metrics["per_class"]["iou"]
    ):
        print(
            f"  Class {class_id}: {iou:.6f}"
        )

    # --------------------------------------------------------
    # SAVE PROFILE
    # --------------------------------------------------------

    profile_path = os.path.join(
        args.output_dir,
        "depth_profile.npy"
    )

    np.save(
        profile_path,
        depth_profile
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    result = {
        "baseline": "B0_depth_majority",
        "train_h5": args.train_h5,
        "test_h5": args.test_h5,
        "n_classes": args.n_classes,
        "train_shape": list(train_labels.shape),
        "test_shape": list(test_labels.shape),
        "pixel_accuracy": metrics["pixel_accuracy"],
        "mean_iou_no_background": metrics[
            "mean_iou_no_background"
        ],
        "per_class_iou": metrics[
            "per_class"
        ]["iou"],
        "depth_profile": depth_profile.tolist(),
    }

    json_path = os.path.join(
        args.output_dir,
        "spatial_baseline_b0.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        f"Profile saved to:\n{profile_path}"
    )

    print(
        f"Metrics saved to:\n{json_path}"
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()