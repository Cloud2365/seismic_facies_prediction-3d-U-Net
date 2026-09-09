"""3D U-Net test with volume-level reconstruction.

The headline test metric is computed on the stitched test cube:

    accumulate softmax probabilities at original voxel coordinates
    -> average overlapping patches
    -> argmax on the full volume
    -> pixel / class / IoU metrics

Patch-level (argmax-then-concat) numbers are still reported for comparison,
but they are NOT the official baseline: overlapping voxels are counted
multiple times.

Test inference uses the same Normalize as training and never applies
RandomRotate / RandomFlip.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))

for candidate in (
    os.path.join(REPO_ROOT, "pytorch-3dunet-master"),
    REPO_ROOT,
    HERE,
):
    if os.path.isdir(os.path.join(candidate, "pytorch3dunet")):
        sys.path.insert(0, candidate)
        break
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pytorch3dunet.datasets.hdf5 import StandardHDF5Dataset
from pytorch3dunet.unet3d.model import get_model

from volume_inference import (
    VolumeAccumulator,
    delta_metrics,
    dump_json,
    expected_overlap_count,
    metrics_from_confusion,
    confusion_from_volumes,
    spatial_slice_from_index,
)


PALETTE = np.array(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Volume-level 3D U-Net test (stitch then argmax)"
    )
    parser.add_argument("--checkpoint", default="./my_models/best_checkpoint_old.pytorch")
    parser.add_argument("--h5", default="./temp_splits/test.h5")
    parser.add_argument("--config", default="./my_configs/generated_config.yaml")
    parser.add_argument("--n_classes", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_vis_samples", type=int, default=8)
    parser.add_argument("--output_dir", default="test_logs_volume")
    parser.add_argument(
        "--save_volume",
        action="store_true",
        help="Save prediction_volume.npy and ground_truth_volume.npy",
    )
    parser.add_argument(
        "--aggregate",
        choices=("prob", "logits"),
        default="prob",
        help="What to average in overlapping voxels (default: softmax probabilities)",
    )
    parser.add_argument("--n_volume_slices", type=int, default=5)
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def collate_patches(batch):
    inputs = torch.stack([item[0] for item in batch], dim=0)
    indices = [item[1] for item in batch]
    return inputs, indices


def normalize_seismic(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr, dtype=np.float32)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val - min_val < 1e-8:
        normalized = np.zeros_like(arr, dtype=np.uint8)
    else:
        normalized = ((arr - min_val) / (max_val - min_val) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(normalized)


def colorize_mask(mask: np.ndarray) -> Image.Image:
    mask = np.asarray(mask).astype(np.int64)
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    n = len(PALETTE)
    for class_id in range(n):
        rgb[mask == class_id] = PALETTE[class_id]
    extra = mask >= n
    if np.any(extra):
        rgb[extra] = np.array([255, 255, 255], dtype=np.uint8)
    return Image.fromarray(rgb)


def hstack_images(images: list[Image.Image], pad: int = 8) -> Image.Image:
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    total_w = sum(widths) + pad * (len(images) - 1)
    total_h = max(heights)
    canvas = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    x = 0
    for im in images:
        if im.mode != "RGB":
            im = im.convert("RGB")
        y = (total_h - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + pad
    return canvas


def save_patch_visualization(sample: dict, patch_index: int, image_root: str) -> None:
    folder = os.path.join(image_root, f"patch_{patch_index:03d}")
    os.makedirs(folder, exist_ok=True)
    raw = sample["raw"]
    if raw.ndim == 4:
        raw = raw[0]
    z_center = raw.shape[0] // 2
    normalize_seismic(raw[z_center]).save(os.path.join(folder, "seismic.png"))
    colorize_mask(sample["gt"][z_center]).save(os.path.join(folder, "ground_truth.png"))
    colorize_mask(sample["pred"][z_center]).save(os.path.join(folder, "prediction.png"))
    sl = sample.get("slice")
    with open(os.path.join(folder, "coords.txt"), "w", encoding="utf-8") as f:
        f.write(f"classes={sample.get('classes', [])}\n")
        f.write(f"slice={sl}\n")


def _axis_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(volume, index, axis=axis)


def pick_indices(length: int, n: int) -> list[int]:
    if length <= 0:
        return []
    n = min(n, length)
    if n == 1:
        return [length // 2]
    return sorted({int(round(i)) for i in np.linspace(0, length - 1, n)})


def save_volume_visualizations(
    raw: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    count: np.ndarray,
    image_root: str,
    n_slices: int,
) -> None:
    os.makedirs(image_root, exist_ok=True)
    views = [
        ("crossline", 0),
        ("depth", 1),
        ("inline", 2),
    ]
    for name, axis in views:
        for idx in pick_indices(raw.shape[axis], n_slices):
            folder = os.path.join(image_root, name, f"idx_{idx:04d}")
            os.makedirs(folder, exist_ok=True)
            seismic = normalize_seismic(_axis_slice(raw, axis, idx))
            truth = colorize_mask(_axis_slice(gt, axis, idx))
            prediction = colorize_mask(_axis_slice(pred, axis, idx))
            seismic.save(os.path.join(folder, "seismic.png"))
            truth.save(os.path.join(folder, "ground_truth.png"))
            prediction.save(os.path.join(folder, "prediction.png"))
            hstack_images([seismic, truth, prediction]).save(
                os.path.join(folder, "comparison.png")
            )
    # coverage of overlapping patches at the volume centre
    z = raw.shape[0] // 2
    coverage = _axis_slice(count, 0, z)
    if coverage.max() > 0:
        cov_img = (coverage / coverage.max() * 255.0).astype(np.uint8)
    else:
        cov_img = np.zeros_like(coverage, dtype=np.uint8)
    Image.fromarray(cov_img).save(
        os.path.join(image_root, f"prediction_count_crossline_{z:04d}.png")
    )


def print_confusion_matrix(cm: np.ndarray, n_classes: int) -> None:
    print()
    print("=" * 120)
    print("NORMALIZED CONFUSION MATRIX (rows = true class, each row sums to 100%)")
    print("=" * 120)
    cm_percent = cm.astype(np.float64)
    row_sums = cm_percent.sum(axis=1, keepdims=True)
    cm_percent = np.divide(
        cm_percent, row_sums, out=np.zeros_like(cm_percent), where=row_sums != 0
    ) * 100.0
    header = f"{'True/Pred':>12}" + "".join(f"{c:>10}" for c in range(n_classes))
    print(header)
    print("-" * 120)
    for true_class in range(n_classes):
        row = f"{true_class:>12}"
        row += "".join(f"{cm_percent[true_class, p]:>9.2f}%" for p in range(n_classes))
        print(row)
    print("-" * 120)
    print()
    print("MAIN CONFUSION PER CLASS")
    for true_class in range(n_classes):
        row = cm_percent[true_class].copy()
        row[true_class] = 0
        ranked = np.argsort(row)[::-1]
        errors = [f"{p} ({row[p]:.2f}%)" for p in ranked[:3] if row[p] > 0]
        if errors:
            print(f"True class {true_class}: " + ", ".join(errors))
        else:
            print(f"True class {true_class}: no errors")


def print_metrics_block(title: str, metrics: dict) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(f"Pixel Accuracy              : {metrics['pixel_accuracy']:.5f}")
    print(f"Mean Class Accuracy         : {metrics['mean_class_accuracy']:.5f}")
    print(f"Mean IoU (no background)    : {metrics['mean_iou_no_background']:.5f}")
    print(f"Mean IoU (with background)  : {metrics['mean_iou_with_background']:.5f}")
    per = metrics["per_class"]
    print()
    print(
        f"{'Class':>8}{'Accuracy':>14}{'IoU':>14}"
        f"{'Precision':>14}{'Recall':>14}{'F1':>14}{'Pixels':>14}"
    )
    print("-" * 100)
    for class_id in range(metrics["n_classes"]):
        print(
            f"{class_id:>8}"
            f"{per['accuracy'][class_id]:>14.4f}"
            f"{per['iou'][class_id]:>14.4f}"
            f"{per['precision'][class_id]:>14.4f}"
            f"{per['recall'][class_id]:>14.4f}"
            f"{per['f1'][class_id]:>14.4f}"
            f"{per['support'][class_id]:>14}"
        )
    print("-" * 100)


def save_metric_tables(output_dir: str, metrics: dict, prefix: str = "") -> None:
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    cm_norm = np.asarray(metrics["confusion_matrix_normalized"], dtype=np.float64)
    stem = prefix + "confusion_matrix"
    np.savetxt(os.path.join(output_dir, stem + ".csv"), cm, delimiter=",", fmt="%d")
    np.savetxt(
        os.path.join(output_dir, stem + "_normalized.csv"),
        cm_norm * 100.0,
        delimiter=",",
        fmt="%.4f",
    )
    per = metrics["per_class"]
    rows = []
    for class_id in range(metrics["n_classes"]):
        rows.append(
            {
                "class": class_id,
                "accuracy": per["accuracy"][class_id],
                "iou": per["iou"][class_id],
                "precision": per["precision"][class_id],
                "recall": per["recall"][class_id],
                "f1": per["f1"][class_id],
                "pixels": per["support"][class_id],
            }
        )
    rows.append(
        {
            "class": "mean_no_bg",
            "accuracy": metrics["mean_class_accuracy_no_background"],
            "iou": metrics["mean_iou_no_background"],
            "precision": metrics["mean_precision_no_background"],
            "recall": metrics["mean_recall_no_background"],
            "f1": metrics["mean_f1_no_background"],
            "pixels": "",
        }
    )
    rows.append(
        {
            "class": "mean_with_bg",
            "accuracy": metrics["mean_class_accuracy"],
            "iou": metrics["mean_iou_with_background"],
            "precision": "",
            "recall": "",
            "f1": "",
            "pixels": "",
        }
    )
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, prefix + "per_class_metrics.csv"),
        index=False,
        float_format="%.6f",
    )


def resolve_slice_builder(config: dict) -> tuple[list[int], list[int], dict]:
    default_patch = [16, 128, 128]
    default_stride = [8, 64, 64]
    loaders = config.get("loaders", {}) if config else {}
    builder = {}
    for phase in ("val", "train", "test"):
        phase_cfg = loaders.get(phase, {})
        if isinstance(phase_cfg, dict) and "slice_builder" in phase_cfg:
            builder = dict(phase_cfg["slice_builder"])
            break
    patch = list(builder.get("patch_shape", default_patch))
    stride = list(builder.get("stride_shape", default_stride))
    builder.setdefault("name", "SliceBuilder")
    builder.setdefault("ndim", 3)
    builder["patch_shape"] = patch
    builder["stride_shape"] = stride
    return patch, stride, builder


def resolve_transformer(config: dict) -> dict:
    """Inference transformer: training Normalize + ToTensor, no augmentations."""
    loaders = config.get("loaders", {}) if config else {}
    for phase in ("val", "test", "train"):
        phase_cfg = loaders.get(phase, {})
        transformer = phase_cfg.get("transformer") if isinstance(phase_cfg, dict) else None
        if transformer:
            raw = [
                step
                for step in transformer.get("raw", [])
                if step.get("name") not in {"RandomRotate", "RandomFlip", "RandomRotate90"}
            ]
            if not any(step.get("name") == "Normalize" for step in raw):
                raw = [{"name": "Normalize"}] + raw
            if not any(step.get("name") == "ToTensor" for step in raw):
                raw = raw + [{"name": "ToTensor", "expand_dims": True}]
            return {
                "raw": raw,
                "label": [
                    {"name": "ToTensor", "dtype": "long", "expand_dims": False}
                ],
            }
    return {
        "raw": [
            {"name": "Normalize"},
            {"name": "ToTensor", "expand_dims": True},
        ],
        "label": [
            {"name": "ToTensor", "dtype": "long", "expand_dims": False}
        ],
    }


def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            "WARNING: checkpoint / config mismatch: "
            f"missing={list(missing)} unexpected={list(unexpected)}"
        )


def ensure_ncdhw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 5:
        return x
    if x.ndim == 4:
        # [B, D, H, W] -> [B, 1, D, H, W]
        return x.unsqueeze(1)
    if x.ndim == 3:
        return x.unsqueeze(0).unsqueeze(0)
    raise RuntimeError(f"Unexpected input ndim={x.ndim}, shape={tuple(x.shape)}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("3D U-NET VOLUME-LEVEL TEST")
    print("=" * 80)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Test H5    : {args.h5}")
    print(f"Config     : {args.config}")
    print(f"Aggregate  : {args.aggregate}")
    print(f"Save volume: {args.save_volume}")

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.isfile(args.h5):
        raise FileNotFoundError(f"Test HDF5 not found: {args.h5}")

    config = load_yaml(args.config)
    model_cfg = dict(config.get("model", {}))
    n_classes = args.n_classes or int(model_cfg.get("out_channels", 10))
    model_cfg.setdefault("in_channels", 1)
    model_cfg.setdefault("out_channels", n_classes)
    model_cfg.setdefault("name", "UNet3D")
    model_cfg.setdefault("is_segmentation", True)

    if model_cfg.get("out_channels") != n_classes:
        raise RuntimeError(
            f"Config out_channels={model_cfg.get('out_channels')} "
            f"!= n_classes={n_classes}"
        )
    if n_classes > 2 and model_cfg.get("final_sigmoid", True):
        print(
            "WARNING: model.final_sigmoid is True/missing while n_classes="
            f"{n_classes}. Training still uses logits for the loss, and "
            "argmax is invariant, but overlapping patches are averaged with "
            "softmax(logits) so the volume-level metric is well-defined."
        )

    patch_shape, stride_shape, slice_builder = resolve_slice_builder(config)
    transformer = resolve_transformer(config)
    print(f"Classes    : {n_classes}")
    print(f"Patch      : {patch_shape}")
    print(f"Stride     : {stride_shape}")
    print(f"Transform  : {[step.get('name') for step in transformer['raw']]}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = get_model(model_cfg)
    load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()
    print("Model loaded.")

    with h5py.File(args.h5, "r") as f:
        if "raw" not in f or "label" not in f:
            raise RuntimeError(f"HDF5 must contain 'raw' and 'label', got {list(f.keys())}")
        raw_volume = f["raw"][:]
        label_volume = np.asarray(f["label"][:])
    if raw_volume.ndim == 4:
        volume_shape = tuple(raw_volume.shape[1:])
        raw_xyz = raw_volume[0]
    else:
        volume_shape = tuple(raw_volume.shape)
        raw_xyz = raw_volume
    if tuple(label_volume.shape[-3:]) != volume_shape:
        raise RuntimeError(
            f"raw spatial shape {volume_shape} != label {label_volume.shape}"
        )
    label_volume = label_volume.reshape(volume_shape)
    print(f"Test volume shape: {volume_shape}")

    dataset = StandardHDF5Dataset(
        file_path=args.h5,
        phase="test",
        slice_builder_config=slice_builder,
        transformer_config=transformer,
    )
    if tuple(dataset.volume_shape) != volume_shape:
        raise RuntimeError(
            f"Dataset volume_shape {dataset.volume_shape} != HDF5 {volume_shape}"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_patches,
    )
    print(f"Number of test patches: {len(dataset)}")

    expected_count = expected_overlap_count(
        volume_shape, tuple(patch_shape), tuple(stride_shape)
    )
    print(
        "Expected overlap count min/max/mean: "
        f"{int(expected_count.min())}/"
        f"{int(expected_count.max())}/"
        f"{float(expected_count.mean()):.3f}"
    )

    accumulator = VolumeAccumulator(volume_shape, n_classes)
    patch_true: list[np.ndarray] = []
    patch_pred: list[np.ndarray] = []
    saved_slices: list[dict[str, Any]] = []
    covered_classes: set[int] = set()

    vis_root = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_root, exist_ok=True)

    print()
    print("=" * 80)
    print("RUNNING TEST  (accumulate probabilities, argmax only after stitch)")
    print("=" * 80)

    with torch.no_grad():
        for input_tensor, indices in tqdm(loader, desc="Testing", unit="batch"):
            input_tensor = ensure_ncdhw(input_tensor).to(device, non_blocking=True)
            _probs, logits = model(input_tensor, return_logits=True)
            if args.aggregate == "logits":
                values = logits
            else:
                values = torch.softmax(logits, dim=1)
            if values.shape[1] != n_classes:
                raise RuntimeError(
                    f"Model output channels {values.shape[1]} != n_classes {n_classes}"
                )
            if values.shape[2:] != tuple(patch_shape) and values.shape[2:] != input_tensor.shape[2:]:
                raise RuntimeError(
                    f"Output spatial shape {tuple(values.shape[2:])} "
                    f"does not match input {tuple(input_tensor.shape[2:])}"
                )
            class_map = torch.argmax(values, dim=1)
            values_np = values.detach().cpu().numpy()
            pred_np = class_map.detach().cpu().numpy()
            raw_np = input_tensor.detach().cpu().numpy()

            for i, index in enumerate(indices):
                sl = spatial_slice_from_index(index)
                accumulator.add_patch(values_np[i], sl)
                gt_patch = np.asarray(label_volume[sl])
                if gt_patch.shape != pred_np[i].shape:
                    raise RuntimeError(
                        f"GT patch {gt_patch.shape} != pred {pred_np[i].shape} at {sl}"
                    )
                patch_true.append(gt_patch.reshape(-1))
                patch_pred.append(pred_np[i].reshape(-1))

                if len(saved_slices) < args.max_vis_samples:
                    gt_classes = set(np.unique(gt_patch).tolist())
                    new_classes = gt_classes - covered_classes
                    if new_classes or len(saved_slices) < min(4, args.max_vis_samples):
                        raw_sample = raw_np[i]
                        if raw_sample.ndim == 4 and raw_sample.shape[0] == 1:
                            raw_sample = raw_sample[0]
                        saved_slices.append(
                            {
                                "raw": raw_sample,
                                "gt": gt_patch,
                                "pred": pred_np[i],
                                "classes": sorted(gt_classes),
                                "slice": sl,
                            }
                        )
                        covered_classes.update(gt_classes)

    recon_stats = accumulator.validate(
        expected_count=expected_count,
        patch_shape=tuple(patch_shape),
        stride_shape=tuple(stride_shape),
    )
    print()
    print("Reconstruction OK")
    print(
        "prediction_count min/max/mean: "
        f"{recon_stats['prediction_count_min']:.3f}/"
        f"{recon_stats['prediction_count_max']:.3f}/"
        f"{recon_stats['prediction_count_mean']:.3f}"
    )

    pred_volume, _ = accumulator.finalize()
    if pred_volume.shape != label_volume.shape:
        raise RuntimeError(
            f"Reconstructed pred {pred_volume.shape} != label {label_volume.shape}"
        )

    volume_cm = confusion_from_volumes(label_volume, pred_volume, n_classes)
    volume_metrics = metrics_from_confusion(volume_cm)

    patch_cm = confusion_from_volumes(
        np.concatenate(patch_true), np.concatenate(patch_pred), n_classes
    )
    patch_metrics = metrics_from_confusion(patch_cm)
    delta = delta_metrics(volume_metrics, patch_metrics)

    print_confusion_matrix(volume_cm, n_classes)
    print_metrics_block(
        "PATCH-LEVEL RESULTS  (legacy, overlapping voxels counted many times)",
        patch_metrics,
    )
    print_metrics_block(
        "VOLUME-LEVEL RESULTS  (official: stitch probabilities, then argmax)",
        volume_metrics,
    )

    print()
    print("=" * 100)
    print("DELTA  (volume - patch)")
    print("=" * 100)
    print(f"Pixel Accuracy              : {delta['pixel_accuracy']:+.5f}")
    print(f"Mean Class Accuracy         : {delta['mean_class_accuracy']:+.5f}")
    print(f"Mean IoU (no background)    : {delta['mean_iou_no_background']:+.5f}")
    print(f"Mean IoU (with background)  : {delta['mean_iou_with_background']:+.5f}")
    print()
    print(f"{'Class':>8}{'dIoU':>14}")
    print("-" * 24)
    for class_id, value in enumerate(delta["per_class_iou"]):
        print(f"{class_id:>8}{value:>+14.4f}")
    print("-" * 24)
    print()
    print(
        "Official baseline is VOLUME-LEVEL mean IoU without background: "
        f"{volume_metrics['mean_iou_no_background']:.5f}"
    )

    payload = {
        "official_metric": "volume_level.mean_iou_no_background",
        "aggregate": args.aggregate,
        "n_classes": n_classes,
        "volume_shape": list(volume_shape),
        "patch_shape": list(patch_shape),
        "stride_shape": list(stride_shape),
        "n_patches": recon_stats["n_patches"],
        "reconstruction": recon_stats,
        "volume_level": volume_metrics,
        "patch_level": patch_metrics,
        "delta_volume_minus_patch": delta,
        "checkpoint": os.path.abspath(args.checkpoint),
        "h5": os.path.abspath(args.h5),
        "config": os.path.abspath(args.config),
        "model": model_cfg,
    }
    dump_json(os.path.join(args.output_dir, "metrics.json"), payload)
    save_metric_tables(args.output_dir, volume_metrics)
    save_metric_tables(args.output_dir, patch_metrics, prefix="patch_level_")

    if args.save_volume:
        pred_path = os.path.join(args.output_dir, "prediction_volume.npy")
        gt_path = os.path.join(args.output_dir, "ground_truth_volume.npy")
        np.save(pred_path, pred_volume.astype(np.uint8, copy=False))
        np.save(gt_path, label_volume.astype(np.uint8, copy=False))
        np.save(
            os.path.join(args.output_dir, "prediction_count.npy"),
            accumulator.prediction_count.astype(np.uint16, copy=False),
        )
        print(f"Saved volumes to {pred_path} and {gt_path}")
    else:
        print("Volumes not saved (pass --save_volume to write npy files)")

    volume_vis_root = os.path.join(args.output_dir, "volume_visualizations")
    save_volume_visualizations(
        raw_xyz,
        label_volume,
        pred_volume,
        accumulator.prediction_count,
        volume_vis_root,
        args.n_volume_slices,
    )
    print(f"Volume visualizations saved to {volume_vis_root}")

    patch_vis_root = os.path.join(vis_root, "patches")
    print(f"Saving {len(saved_slices)} patch visualizations")
    for idx, sample in enumerate(saved_slices):
        save_patch_visualization(sample, idx, patch_vis_root)

    missing = set(range(n_classes)) - covered_classes
    if missing:
        print(f"WARNING: patch visualizations missed classes {sorted(missing)}")

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
