import argparse
import os
import sys

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from patch_boundary_metrics import PatchBoundaryStats

# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, PROJECT_ROOT)

from volume_inference import (
    VolumeAccumulator,
    confusion_from_volumes,
    expected_overlap_count,
    metrics_from_confusion,
    spatial_slice_from_index,
)

from pytorch3dunet.datasets.hdf5 import (
    StandardHDF5Dataset,
    LazyHDF5DatasetWithDepth,
)
from monotonic_decode import monotonic_decode
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.config import load_config

# ============================================================
# ARGUMENTS
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description="3D U-Net test with confusion matrix and visualization"
    )

    parser.add_argument(
        "--checkpoint",
        default="./my_models/best_checkpoint_batch8_noNormalize.pytorch",
        help="Path to checkpoint.pytorch",
    )

    parser.add_argument(
        "--h5", default="temp_splits/spatial/test.h5", help="Path to test HDF5 file"
    )

    parser.add_argument(
        "--config",
        default="./my_configs/generated_config_spatial.yaml",
        help="Path to training YAML config",
    )

    parser.add_argument(
        "--n_classes",
        type=int,
        default=None,
        help="Number of classes (default: model.out_channels from the config)",
    )

    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for testing"
    )

    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of DataLoader workers"
    )

    parser.add_argument(
        "--max_vis_samples",
        type=int,
        default=16,
        help="Maximum number of patches saved for visualization",
    )

    parser.add_argument(
        "--output_dir",
        default="test_logs/spatial/spatial_batch_8",
        help="Directory for test results",
    )
    parser.add_argument(
        "--absolute_depth",
        action="store_true",
        help="Deprecated: detected automatically from the config "
        "(loaders.dataset / model.in_channels). Kept for old command lines.",
    )
    parser.add_argument(
        "--monotonic",
        action="store_true",
        help="Also decode labels so that they never decrease with depth on every trace "
        "(classes 0..9 are stratigraphically ordered top to bottom). "
        "Metrics are reported for both plain argmax and monotonic decoding.",
    )

    return parser.parse_args()


# ============================================================
# NORMALIZE SEISMIC
# ============================================================


def normalize_seismic(arr):

    arr = np.asarray(arr, dtype=np.float32)

    min_val = np.min(arr)
    max_val = np.max(arr)

    if max_val - min_val < 1e-8:
        normalized = np.zeros_like(arr)
    else:
        normalized = (arr - min_val) / (max_val - min_val) * 255.0

    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    return Image.fromarray(normalized)


# ============================================================
# COLORIZE MASK
# ============================================================


def colorize_mask(mask):

    mask = np.asarray(mask).astype(np.uint8)

    # 10 классов
    palette = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 0],
            [128, 0, 255],
            [128, 128, 128],
        ],
        dtype=np.uint8,
    )

    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)

    for class_id in range(len(palette)):

        rgb[mask == class_id] = palette[class_id]

    return Image.fromarray(rgb)


# ============================================================
# SAVE VISUALIZATION
# ============================================================


def save_patch_visualization(sample, patch_index, image_root):

    folder = os.path.join(image_root, f"patch_{patch_index:03d}")

    os.makedirs(folder, exist_ok=True)

    raw = sample["raw"]

    # raw обычно [C, Z, Y, X]
    if raw.ndim == 4:
        raw = raw[0]

    gt = sample["gt"]
    pred = sample["pred"]

    classes = sample.get("classes", [])

    print(f"Patch {patch_index:03d}: " f"classes={classes}")

    # --------------------------------------------------------
    # Центральный Z-срез
    # --------------------------------------------------------

    z_center = raw.shape[0] // 2

    raw_view = raw[z_center, :, :]

    gt_view = gt[z_center, :, :]

    pred_view = pred[z_center, :, :]

    # --------------------------------------------------------
    # Сохраняем 3 изображения прямо в patch_XXX
    # --------------------------------------------------------

    normalize_seismic(raw_view).save(os.path.join(folder, "seismic.png"))

    colorize_mask(gt_view).save(os.path.join(folder, "ground_truth.png"))

    colorize_mask(pred_view).save(os.path.join(folder, "prediction.png"))


# ============================================================
# PRINT CONFUSION MATRIX
# ============================================================


def print_confusion_matrix(cm, n_classes):

    print()
    print("=" * 120)
    print("CONFUSION MATRIX — PERCENTAGES")
    print("Rows = TRUE CLASS")
    print("Columns = PREDICTED CLASS")
    print("Each row is normalized to 100%")
    print("=" * 120)

    cm_percent = cm.astype(np.float64)

    row_sums = cm_percent.sum(axis=1, keepdims=True)

    cm_percent = (
        np.divide(
            cm_percent, row_sums, out=np.zeros_like(cm_percent), where=row_sums != 0
        )
        * 100.0
    )

    print(f"{'True/Pred':>12}", end="")

    for class_id in range(n_classes):

        print(f"{class_id:>10}", end="")

    print()

    print("-" * 120)

    for true_class in range(n_classes):

        print(f"{true_class:>12}", end="")

        for pred_class in range(n_classes):

            print(f"{cm_percent[true_class, pred_class]:>9.2f}%", end="")

        print()

    print("-" * 120)

    # --------------------------------------------------------
    # Главные ошибки каждого класса
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("MAIN CONFUSION PER CLASS")
    print("=" * 100)

    for true_class in range(n_classes):

        row = cm_percent[true_class].copy()

        # Убираем правильное предсказание
        row[true_class] = 0

        sorted_indices = np.argsort(row)[::-1]

        errors = []

        for pred_class in sorted_indices[:3]:

            if row[pred_class] > 0:

                errors.append(f"{pred_class} " f"({row[pred_class]:.2f}%)")

        if errors:

            print(f"True class {true_class}: " + ", ".join(errors))

        else:

            print(f"True class {true_class}: " "no errors")

    print("=" * 100)


# ============================================================
# DATASET HELPERS
# ============================================================


def test_raw_transforms(val_raw_cfg, use_depth):
    """Raw transforms for the test: the validation transforms of the training config.

    Only deterministic transforms are kept. Old configs are handled for compatibility:
      * a 1-channel config without Normalize keeps the old hard-coded per-patch Normalize;
      * a depth config without Normalize falls back to per-patch normalization inside
        LazyHDF5DatasetWithDepth (exactly what those checkpoints were trained with).
    """
    allowed = {"Normalize", "Standardize", "PercentileNormalizer", "ToTensor"}
    transforms = [dict(t) for t in val_raw_cfg if t.get("name") in allowed]
    names = [t["name"] for t in transforms]
    if not use_depth and not ({"Normalize", "Standardize", "PercentileNormalizer"} & set(names)):
        transforms.insert(0, {"name": "Normalize"})
    transforms = [t for t in transforms if t["name"] != "ToTensor"]
    transforms.append({"name": "ToTensor", "expand_dims": True})
    return transforms


def collate_test_batch(batch):
    """Stack patches; keep slice indices as a plain list (default_collate cannot batch slices)."""
    raws = [torch.as_tensor(item[0]) for item in batch]
    raws = [r.unsqueeze(0) if r.ndim == 3 else r for r in raws]
    return torch.stack(raws, dim=0), [item[1] for item in batch]


# ============================================================
# MAIN
# ============================================================


def main():

    args = parse_args()

    print()
    print("=" * 80)
    print("3D U-NET TEST")
    print("=" * 80)

    print(f"Checkpoint : {args.checkpoint}")

    print(f"Test H5    : {args.h5}")

    print(f"Config     : {args.config}")

    print(f"Batch size : {args.batch_size}")

    print(f"Visualize  : {args.max_vis_samples}")

    print("=" * 80)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # --------------------------------------------------------
    # LOAD CONFIG
    # --------------------------------------------------------

    print()
    print("Loading config...")

    # load_config обычно ожидает sys.argv.
    # Поэтому временно подменяем argv.
    old_argv = sys.argv

    sys.argv = ["test.py", "--config", args.config]

    try:

        config, _ = load_config()

    finally:

        sys.argv = old_argv

    if args.n_classes is None:
        args.n_classes = int(config["model"]["out_channels"])
    elif args.n_classes != int(config["model"]["out_channels"]):
        raise ValueError(
            f"--n_classes={args.n_classes} does not match model.out_channels="
            f"{config['model']['out_channels']} in {args.config}"
        )
    print(f"Classes    : {args.n_classes}")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("Creating model...")

    model = get_model(config["model"])

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

        model.load_state_dict(checkpoint["model_state_dict"])

    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:

        model.load_state_dict(checkpoint["state_dict"])

    else:

        model.load_state_dict(checkpoint)

    model = model.to(device)

    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # HDF5 CHECK
    # --------------------------------------------------------

    print()
    print("Checking HDF5...")

    with h5py.File(args.h5, "r") as f:

        print("Datasets in HDF5:")

        for key in f.keys():

            print(f"  - {key}")

        if "raw" not in f:

            raise RuntimeError("Dataset 'raw' not found in HDF5.")

        if "label" not in f:

            raise RuntimeError("Dataset 'label' not found in HDF5.")

    # --------------------------------------------------------
    # DATASET (everything is taken from the training config)
    # --------------------------------------------------------

    print()
    print("Creating test dataset...")

    loaders_cfg = config["loaders"]
    val_cfg = loaders_cfg["val"]
    patch_shape = tuple(val_cfg["slice_builder"]["patch_shape"])
    stride_shape = tuple(val_cfg["slice_builder"]["stride_shape"])
    use_depth = (
        loaders_cfg.get("dataset") == "LazyHDF5DatasetWithDepth"
        or config["model"].get("in_channels", 1) == 2
    )
    if args.absolute_depth and not use_depth:
        raise ValueError("--absolute_depth given, but the config describes a 1-channel model")

    raw_transforms = test_raw_transforms(val_cfg["transformer"]["raw"], use_depth)
    transformer_config = {
        "raw": raw_transforms,
        "label": [{"name": "ToTensor", "dtype": "long", "expand_dims": False}],
    }
    slice_builder_config = {
        "name": "SliceBuilder",
        "ndim": 3,
        "patch_shape": list(patch_shape),
        "stride_shape": list(stride_shape),
    }

    print(f"Patch shape : {patch_shape}")
    print(f"Stride      : {stride_shape}")
    print(f"Depth input : {use_depth}")
    print(f"Raw transforms: {raw_transforms}")

    dataset_cls = LazyHDF5DatasetWithDepth if use_depth else StandardHDF5Dataset
    dataset = dataset_cls(
        file_path=args.h5,
        phase="test",
        slice_builder_config=slice_builder_config,
        transformer_config=transformer_config,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_test_batch,
    )

    print(f"Number of test patches: " f"{len(dataset)}")

    # --------------------------------------------------------
    # OUTPUT DIRECTORIES
    # --------------------------------------------------------

    image_root = os.path.join(args.output_dir, "visualizations")

    os.makedirs(image_root, exist_ok=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------------
    # METRIC STORAGE
    # --------------------------------------------------------

    expected_count = expected_overlap_count(dataset.volume_shape, patch_shape, stride_shape)

    accumulator = VolumeAccumulator(dataset.volume_shape, args.n_classes)

    saved_slices = []

    covered_classes = set()

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RUNNING TEST")
    print("=" * 80)
    # border 4 as before, reduced for thin patches (the stats need 2*border < min patch size)
    boundary_border = max(1, min(4, (min(patch_shape) - 1) // 2))
    boundary_stats = PatchBoundaryStats(n_classes=args.n_classes, border=boundary_border)

    with h5py.File(args.h5, "r") as test_h5, torch.no_grad():
        label_volume = np.asarray(test_h5["label"][:])

        for input_tensor, patch_slices in tqdm(
            loader, desc="Testing", total=len(loader), unit="batch"
        ):
            # [B, C, D, H, W]
            input_tensor = input_tensor.to(device, non_blocking=True)

            _probs, logits = model(input_tensor, return_logits=True)
            values = torch.softmax(logits, dim=1)
            prediction = torch.argmax(values, dim=1)

            values_np = values.cpu().numpy()
            pred_np = prediction.cpu().numpy()

            for sample_idx, patch_slice in enumerate(patch_slices):
                sl = spatial_slice_from_index(patch_slice)
                label_np = label_volume[sl]

                boundary_stats.add_patch(pred_np[sample_idx], label_np)
                accumulator.add_patch(values_np[sample_idx], sl)

                # ------------------------------------------------
                # VISUALIZATION SELECTION
                # ------------------------------------------------
                if len(saved_slices) >= args.max_vis_samples:
                    continue

                patch_classes = set(np.unique(label_np).tolist())
                new_classes = patch_classes - covered_classes
                should_save = bool(new_classes) or len(saved_slices) < min(
                    10, args.max_vis_samples
                )
                if should_save:
                    raw_sample = input_tensor[sample_idx, 0].cpu().numpy()
                    saved_slices.append(
                        {
                            "raw": raw_sample,
                            "gt": np.asarray(label_np),
                            "pred": pred_np[sample_idx],
                            "classes": sorted(patch_classes),
                            "slice": patch_slice,
                        }
                    )
                    covered_classes.update(patch_classes)

    # --------------------------------------------------------
    # VOLUME-LEVEL RECONSTRUCTION
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RECONSTRUCTING TEST VOLUME")
    print("=" * 80)

    recon_stats = accumulator.validate(
        expected_count=expected_count,
        patch_shape=patch_shape,
        stride_shape=stride_shape,
    )

    print("Reconstruction OK")

    print(
        f"prediction_count min/max/mean: "
        f"{recon_stats['prediction_count_min']:.3f}/"
        f"{recon_stats['prediction_count_max']:.3f}/"
        f"{recon_stats['prediction_count_mean']:.3f}"
    )

    # --------------------------------------------------------
    # FINALIZE FULL VOLUME
    # --------------------------------------------------------

    pred_volume, averaged = accumulator.finalize()

    argmax_metrics = None
    if args.monotonic:
        argmax_metrics = metrics_from_confusion(
            confusion_from_volumes(label_volume, pred_volume, args.n_classes)
        )
        np.save(os.path.join(args.output_dir, "prediction_volume_argmax.npy"), pred_volume)
        print("Monotonic decoding along depth...")
        pred_volume = monotonic_decode(averaged).astype(pred_volume.dtype, copy=False)
        print(
            "Argmax     mIoU (without class 0): "
            f"{argmax_metrics['mean_iou_no_background']:.5f}"
        )
    # Сохраняем reconstructed full volume и карту перекрытий патчей
    np.save(os.path.join(args.output_dir, "prediction_volume.npy"), pred_volume)

    np.save(os.path.join(args.output_dir, "ground_truth_volume.npy"), label_volume)

    np.save(
        os.path.join(args.output_dir, "prediction_count.npy"),
        accumulator.prediction_count.astype(np.uint8),
    )

    print("Saved prediction_volume.npy")
    print("Saved ground_truth_volume.npy")
    print("Saved prediction_count.npy")

    confidence = averaged.max(axis=0).astype(np.float16)

    entropy = np.zeros(averaged.shape[1:], dtype=np.float32)

    buffer = np.empty(averaged.shape[1:], dtype=np.float32)

    for class_id in range(averaged.shape[0]):

        p = averaged[class_id]

        # clip(p, 1e-8, 1.0) без создания отдельного массива
        np.maximum(p, 1e-8, out=buffer)

        np.minimum(buffer, 1.0, out=buffer)

        # buffer = log(p)
        np.log(buffer, out=buffer)

        # buffer = p * log(p)
        np.multiply(p, buffer, out=buffer)

        # H = -sum(p * log(p))
        entropy -= buffer

    entropy = entropy.astype(np.float16)

    del buffer

    # --------------------------------------------------------
    # VOLUME-LEVEL CONFUSION MATRIX
    # --------------------------------------------------------

    volume_cm = confusion_from_volumes(label_volume, pred_volume, args.n_classes)

    # --------------------------------------------------------
    # VOLUME-LEVEL METRICS
    # --------------------------------------------------------

    metrics = metrics_from_confusion(volume_cm)

    pixel_accuracy = metrics["pixel_accuracy"]

    mean_class_accuracy = metrics["mean_class_accuracy"]

    mean_iou_no_background = metrics["mean_iou_no_background"]

    mean_iou_with_background = metrics["mean_iou_with_background"]

    per_class = metrics["per_class"]

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("FINAL TEST RESULTS — VOLUME LEVEL")
    print("=" * 100)

    print(f"Pixel Accuracy              : " f"{pixel_accuracy:.5f}")

    print(f"Mean Class Accuracy         : " f"{mean_class_accuracy:.5f}")

    print(f"Mean IoU (without class 0)  : " f"{mean_iou_no_background:.5f}")

    print(f"Mean IoU (with class 0)     : " f"{mean_iou_with_background:.5f}")

    if argmax_metrics is not None:
        print()
        print("Decoding: monotonic along depth (metrics above)")
        print(
            "Plain argmax mIoU (without class 0): "
            f"{argmax_metrics['mean_iou_no_background']:.5f}, "
            f"pixel accuracy: {argmax_metrics['pixel_accuracy']:.5f}"
        )

    print()
    print(
        f"{'Class':>8}"
        f"{'Accuracy':>14}"
        f"{'IoU':>14}"
        f"{'Precision':>14}"
        f"{'Recall':>14}"
        f"{'F1':>14}"
        f"{'Support':>14}"
    )

    print("-" * 100)

    # --------------------------------------------------------
    # PER-CLASS METRICS
    # --------------------------------------------------------

    for class_id in range(args.n_classes):

        class_metrics = {
            "accuracy": per_class["accuracy"][class_id],
            "iou": per_class["iou"][class_id],
            "precision": per_class["precision"][class_id],
            "recall": per_class["recall"][class_id],
            "f1": per_class["f1"][class_id],
            "support": per_class["support"][class_id],
        }

        print(
            f"{class_id:>8}"
            f"{class_metrics['accuracy']:>14.4f}"
            f"{class_metrics['iou']:>14.4f}"
            f"{class_metrics['precision']:>14.4f}"
            f"{class_metrics['recall']:>14.4f}"
            f"{class_metrics['f1']:>14.4f}"
            f"{class_metrics['support']:>14}"
        )

    print("-" * 100)

    # --------------------------------------------------------
    # CONFUSION MATRIX — CONSOLE
    # --------------------------------------------------------

    print_confusion_matrix(volume_cm, args.n_classes)

    # --------------------------------------------------------
    # SAVE CONFUSION MATRIX CSV
    # --------------------------------------------------------

    cm_percent = volume_cm.astype(np.float64)

    row_sums = cm_percent.sum(axis=1, keepdims=True)

    cm_percent = (
        np.divide(
            cm_percent, row_sums, out=np.zeros_like(cm_percent), where=row_sums != 0
        )
        * 100.0
    )

    cm_csv = os.path.join(args.output_dir, "confusion_matrix_percent.csv")

    np.savetxt(cm_csv, cm_percent, delimiter=",", fmt="%.4f")

    print()
    print(f"Confusion matrix saved to:" f"\n{cm_csv}")

    # --------------------------------------------------------
    # SAVE CLASS METRICS CSV
    # --------------------------------------------------------

    results_csv = os.path.join(args.output_dir, "class_metrics.csv")

    rows = []

    for class_id in range(args.n_classes):

        rows.append(
            {
                "class": class_id,
                "accuracy": per_class["accuracy"][class_id],
                "iou": per_class["iou"][class_id],
                "precision": per_class["precision"][class_id],
                "recall": per_class["recall"][class_id],
                "f1": per_class["f1"][class_id],
                "pixels": per_class["support"][class_id],
            }
        )

    # --------------------------------------------------------
    # MEAN ROW
    # --------------------------------------------------------

    rows.append(
        {
            "class": "Mean",
            "accuracy": mean_class_accuracy,
            "iou": mean_iou_no_background,
            "precision": np.mean(
                [per_class["precision"][class_id] for class_id in range(args.n_classes)]
            ),
            "recall": np.mean(
                [per_class["recall"][class_id] for class_id in range(args.n_classes)]
            ),
            "f1": np.mean(
                [per_class["f1"][class_id] for class_id in range(args.n_classes)]
            ),
            "pixels": "",
        }
    )

    results_df = pd.DataFrame(rows)

    results_df.to_csv(results_csv, index=False, float_format="%.6f")

    print(f"Class metrics saved to:" f"\n{results_csv}")
    confidence_path = os.path.join(args.output_dir, "prediction_confidence.npy")

    entropy_path = os.path.join(args.output_dir, "prediction_entropy.npy")

    np.save(confidence_path, confidence)

    np.save(entropy_path, entropy)

    print(f"Confidence map saved to:\n{confidence_path}")

    print(f"Entropy map saved to:\n{entropy_path}")
    boundary_path = os.path.join(args.output_dir, "patch_boundary_metrics.csv")
    boundary_stats.save_csv(boundary_path)
    print(boundary_stats.summary_line())
    print(boundary_path)
    # --------------------------------------------------------
    # SAVE VISUALIZATIONS
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(f"SAVING {len(saved_slices)} " f"VISUALIZATION PATCHES")
    print("=" * 80)

    for idx, sample in enumerate(saved_slices):

        save_patch_visualization(sample, idx, image_root)

    print()
    print(f"Images saved to:" f"\n{image_root}")

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("VISUALIZATION CLASS COVERAGE")
    print("=" * 80)

    print("Classes covered by selected patches:")

    print(sorted(covered_classes))

    missing_classes = set(range(args.n_classes)) - covered_classes

    if missing_classes:

        print("WARNING: classes not covered:")

        print(sorted(missing_classes))

    else:

        print("All classes are represented " "in selected patches.")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
