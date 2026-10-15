from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


# ============================================================
# PROJECT ROOT / IMPORTS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from source.volume_inference import (
    confusion_from_volumes,
    metrics_from_confusion,
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnostics for spatial shift in seismic facies"
    )

    parser.add_argument(
        "--train_h5",
        type=str,
        default="./temp_splits/spatial/train.h5",
    )

    parser.add_argument(
        "--test_h5",
        type=str,
        default="./temp_splits/spatial/test.h5",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./spatial_shift_diagnostics",
    )

    parser.add_argument(
        "--n_classes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--raw_sample_step",
        type=int,
        default=4,
        help="Subsampling step for amplitude quantiles",
    )

    return parser.parse_args()


# ============================================================
# DEPTH BASELINE
# ============================================================

def build_depth_profile(
    train_labels: np.ndarray,
    n_classes: int,
) -> np.ndarray:

    if train_labels.ndim != 3:
        raise ValueError(
            f"Expected train labels [crossline, depth, inline], "
            f"got {train_labels.shape}"
        )

    n_depth = train_labels.shape[1]

    profile = np.empty(
        n_depth,
        dtype=np.int64,
    )

    for d in range(n_depth):
        values = train_labels[:, d, :].reshape(-1)

        counts = np.bincount(
            values,
            minlength=n_classes,
        )

        profile[d] = np.argmax(counts)

    return profile


def quartile_slices(n_crosslines: int):
    edges = np.linspace(
        0,
        n_crosslines,
        5,
        dtype=int,
    )

    return {
        "Q1": slice(edges[0], edges[1]),
        "Q2": slice(edges[1], edges[2]),
        "Q3": slice(edges[2], edges[3]),
        "Q4": slice(edges[3], edges[4]),
    }


def evaluate_prediction(
    gt: np.ndarray,
    pred: np.ndarray,
):
    cm = confusion_from_volumes(
        gt,
        pred,
        10,
    )

    return metrics_from_confusion(cm)


# ============================================================
# B0 BY QUARTILES
# ============================================================

def analyze_b0_quartiles(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    n_classes: int,
):

    profile = build_depth_profile(
        train_labels,
        n_classes,
    )

    prediction = np.broadcast_to(
        profile[None, :, None],
        test_labels.shape,
    ).astype(np.int64)

    quartiles = quartile_slices(
        test_labels.shape[0]
    )

    result = {}

    for name, sl in quartiles.items():

        gt_q = test_labels[sl]
        pred_q = prediction[sl]

        cm = confusion_from_volumes(
            gt_q,
            pred_q,
            n_classes,
        )

        metrics = metrics_from_confusion(cm)

        result[name] = {
            "crossline_start": int(sl.start),
            "crossline_end": int(sl.stop - 1),
            "n_voxels": int(gt_q.size),
            "pixel_accuracy": float(
                metrics["pixel_accuracy"]
            ),
            "mean_iou_no_background": float(
                metrics["mean_iou_no_background"]
            ),
            "per_class_iou": [
                float(x)
                for x in metrics["per_class"]["iou"]
            ],
        }

    return profile, result


# ============================================================
# CLASS DEPTH STATISTICS
# ============================================================

def class_depth_stats(
    labels: np.ndarray,
    classes=(3, 4, 5, 6),
):
    """
    Axis 1 = absolute depth Y.
    """

    result = {}

    for cls in classes:

        mask = labels == cls

        positions = np.where(mask)[1]

        if positions.size == 0:
            result[str(cls)] = {
                "n": 0,
                "mean_y": None,
                "median_y": None,
                "std_y": None,
            }

            continue

        result[str(cls)] = {
            "n": int(positions.size),
            "mean_y": float(
                positions.mean()
            ),
            "median_y": float(
                np.median(positions)
            ),
            "std_y": float(
                positions.std()
            ),
        }

    return result


# ============================================================
# AMPLITUDE STATISTICS
# ============================================================

def amplitude_stats(
    h5_path: str,
    crossline_slice: slice,
    sample_step: int = 4,
):
    """
    Calculates exact mean/std using all sampled crosslines.

    Quantiles are calculated on a deterministic subsample to
    avoid loading the complete raw cube into RAM.
    """

    with h5py.File(h5_path, "r") as f:

        raw = f["raw"]

        start = crossline_slice.start or 0
        stop = crossline_slice.stop

        if stop is None:
            stop = raw.shape[0]

        sum_x = 0.0
        sum_x2 = 0.0
        n = 0

        samples = []

        for z0 in range(
            start,
            stop,
            sample_step,
        ):

            x = np.asarray(
                raw[z0],
                dtype=np.float32,
            )

            flat = x.reshape(-1)

            sum_x += float(
                flat.sum(dtype=np.float64)
            )

            sum_x2 += float(
                np.square(flat, dtype=np.float64).sum(
                    dtype=np.float64
                )
            )

            n += flat.size

            samples.append(flat[::sample_step])

    mean = sum_x / n

    variance = max(
        sum_x2 / n - mean * mean,
        0.0,
    )

    std = np.sqrt(variance)

    if samples:
        sample_values = np.concatenate(samples)
    else:
        sample_values = np.empty(
            0,
            dtype=np.float32,
        )

    if sample_values.size:

        quantiles = np.quantile(
            sample_values,
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
        )

        min_value = float(
            sample_values.min()
        )

        max_value = float(
            sample_values.max()
        )

    else:

        quantiles = np.full(
            7,
            np.nan,
        )

        min_value = None
        max_value = None

    return {
        "n_exact": int(n),
        "sample_n": int(sample_values.size),
        "mean": float(mean),
        "std": float(std),
        "min_sample": min_value,
        "max_sample": max_value,
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "q50": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
    }


# ============================================================
# AMPLITUDE BY DEPTH
# ============================================================

def amplitude_by_depth(
    h5_path: str,
    crossline_slice: slice,
    sample_step: int = 4,
):
    """
    Raw shape:
        [crossline, depth, inline]

    Для каждой глубины считаем mean/std амплитуды
    по выбранным crosslines и inline-позициям.
    """

    with h5py.File(h5_path, "r") as f:

        raw = f["raw"]

        start = crossline_slice.start or 0
        stop = crossline_slice.stop

        if stop is None:
            stop = raw.shape[0]

        depth_n = raw.shape[1]

        sums = np.zeros(
            depth_n,
            dtype=np.float64,
        )

        sums_sq = np.zeros(
            depth_n,
            dtype=np.float64,
        )

        counts = np.zeros(
            depth_n,
            dtype=np.int64,
        )

        for z0 in range(
            start,
            stop,
            sample_step,
        ):

            # [depth, inline]
            x = np.asarray(
                raw[
                    z0,
                    :,
                    ::sample_step,
                ],
                dtype=np.float32,
            )

            sums += x.sum(
                axis=1,
                dtype=np.float64,
            )

            sums_sq += np.square(
                x,
                dtype=np.float64,
            ).sum(
                axis=1,
                dtype=np.float64,
            )

            counts += x.shape[1]

    means = sums / counts

    variances = (
        sums_sq / counts
        - means * means
    )

    variances = np.maximum(
        variances,
        0.0,
    )

    stds = np.sqrt(
        variances
    )

    return means, stds
# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    out = Path(args.output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("SPATIAL SHIFT DIAGNOSTICS")
    print("=" * 80)

    print()
    print("Train:", args.train_h5)
    print("Test :", args.test_h5)

    # --------------------------------------------------------
    # LOAD LABELS
    # --------------------------------------------------------

    with h5py.File(args.train_h5, "r") as f:
        train_labels = np.asarray(
            f["label"][:]
        )

    with h5py.File(args.test_h5, "r") as f:
        test_labels = np.asarray(
            f["label"][:]
        )

    print()
    print(
        "Train labels:",
        train_labels.shape,
    )

    print(
        "Test labels :",
        test_labels.shape,
    )

    # --------------------------------------------------------
    # QUARTILES
    # --------------------------------------------------------

    quartiles = quartile_slices(
        test_labels.shape[0]
    )

    # --------------------------------------------------------
    # B0 QUARTILES
    # --------------------------------------------------------

    profile, b0_quartiles = analyze_b0_quartiles(
        train_labels,
        test_labels,
        args.n_classes,
    )

    print()
    print("=" * 80)
    print("B0 DEPTH-ONLY BY QUARTILE")
    print("=" * 80)

    for name, values in b0_quartiles.items():

        print(
            f"{name}: "
            f"mIoU={values['mean_iou_no_background']:.6f}, "
            f"PA={values['pixel_accuracy']:.6f}"
        )

    # --------------------------------------------------------
    # CLASS DEPTH STATISTICS
    # --------------------------------------------------------

    train_depth = class_depth_stats(
        train_labels
    )

    test_depth = class_depth_stats(
        test_labels
    )

    quartile_depth = {}

    for name, sl in quartiles.items():

        quartile_depth[name] = class_depth_stats(
            test_labels[sl]
        )

    # --------------------------------------------------------
    # AMPLITUDE STATS
    # --------------------------------------------------------

    amplitude_regions = {
        "train": (
            args.train_h5,
            slice(0, train_labels.shape[0]),
        ),
        "test_Q1": (
            args.test_h5,
            quartiles["Q1"],
        ),
        "test_Q2": (
            args.test_h5,
            quartiles["Q2"],
        ),
        "test_Q3": (
            args.test_h5,
            quartiles["Q3"],
        ),
        "test_Q4": (
            args.test_h5,
            quartiles["Q4"],
        ),
    }

    amplitude = {}

    print()
    print("=" * 80)
    print("AMPLITUDE STATISTICS")
    print("=" * 80)

    for name, (
        h5_path,
        sl,
    ) in amplitude_regions.items():

        print(
            f"\n{name}:"
        )

        stats = amplitude_stats(
            h5_path,
            sl,
            args.raw_sample_step,
        )

        amplitude[name] = stats

        print(
            f"  mean = {stats['mean']:.6f}"
        )

        print(
            f"  std  = {stats['std']:.6f}"
        )

        print(
            f"  q01  = {stats['q01']:.6f}"
        )

        print(
            f"  q50  = {stats['q50']:.6f}"
        )

        print(
            f"  q99  = {stats['q99']:.6f}"
        )

    # --------------------------------------------------------
    # AMPLITUDE BY DEPTH
    # --------------------------------------------------------

    print()
    print(
        "Calculating amplitude profiles by depth..."
    )

    amp_depth_train = amplitude_by_depth(
        args.train_h5,
        slice(0, train_labels.shape[0]),
        args.raw_sample_step,
    )

    amp_depth_q1 = amplitude_by_depth(
        args.test_h5,
        quartiles["Q1"],
        args.raw_sample_step,
    )

    amp_depth_q4 = amplitude_by_depth(
        args.test_h5,
        quartiles["Q4"],
        args.raw_sample_step,
    )

    # --------------------------------------------------------
    # SAVE DEPTH PROFILE
    # --------------------------------------------------------

    np.save(
        out / "b0_depth_profile.npy",
        profile,
    )

    # --------------------------------------------------------
    # SAVE AMPLITUDE PROFILE CSV
    # --------------------------------------------------------

    amplitude_csv = out / "amplitude_by_depth.csv"

    with amplitude_csv.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "depth,"
            "train_mean,train_std,"
            "q1_mean,q1_std,"
            "q4_mean,q4_std\n"
        )

        for d in range(
            len(amp_depth_train[0])
        ):

            f.write(
                f"{d},"
                f"{amp_depth_train[0][d]:.8f},"
                f"{amp_depth_train[1][d]:.8f},"
                f"{amp_depth_q1[0][d]:.8f},"
                f"{amp_depth_q1[1][d]:.8f},"
                f"{amp_depth_q4[0][d]:.8f},"
                f"{amp_depth_q4[1][d]:.8f}\n"
            )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    result = {
        "train_shape": list(
            train_labels.shape
        ),
        "test_shape": list(
            test_labels.shape
        ),
        "b0_depth_profile": profile.tolist(),
        "b0_quartiles": b0_quartiles,
        "class_depth": {
            "train": train_depth,
            "test": test_depth,
            "test_quartiles": quartile_depth,
        },
        "amplitude": amplitude,
        "amplitude_by_depth": {
            "train": {
                "mean": amp_depth_train[0].tolist(),
                "std": amp_depth_train[1].tolist(),
            },
            "test_Q1": {
                "mean": amp_depth_q1[0].tolist(),
                "std": amp_depth_q1[1].tolist(),
            },
            "test_Q4": {
                "mean": amp_depth_q4[0].tolist(),
                "std": amp_depth_q4[1].tolist(),
            },
        },
    }

    with (
        out / "spatial_shift_diagnostics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        "\nSaved:"
    )

    print(
        out / "spatial_shift_diagnostics.json"
    )

    print(
        out / "amplitude_by_depth.csv"
    )

    print(
        out / "b0_depth_profile.npy"
    )


if __name__ == "__main__":
    main()