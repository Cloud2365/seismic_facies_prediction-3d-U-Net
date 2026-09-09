import argparse
import os
import sys

import h5py
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from pytorch3dunet.datasets.hdf5 import StandardHDF5Dataset
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
        default="./my_models/best_checkpoint_old.pytorch",
        help="Path to checkpoint.pytorch"
    )

    parser.add_argument(
        "--h5",
        default="./temp_splits/test.h5",
        help="Path to test HDF5 file"
    )

    parser.add_argument(
        "--config",
        default="./my_configs/generated_config.yaml",
        help="Path to training YAML config"
    )

    parser.add_argument(
        "--n_classes",
        type=int,
        default=10,
        help="Number of classes"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for testing"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers"
    )

    parser.add_argument(
        "--max_vis_samples",
        type=int,
        default=20,
        help="Maximum number of patches saved for visualization"
    )

    parser.add_argument(
        "--output_dir",
        default="test_logs_old",
        help="Directory for test results"
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
        normalized = (
            (arr - min_val)
            / (max_val - min_val)
            * 255.0
        )

    normalized = np.clip(
        normalized,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(normalized)


# ============================================================
# COLORIZE MASK
# ============================================================

def colorize_mask(mask):

    mask = np.asarray(mask).astype(np.uint8)

    # 10 классов
    palette = np.array([
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
    ], dtype=np.uint8)

    rgb = np.zeros(
        mask.shape + (3,),
        dtype=np.uint8
    )

    for class_id in range(len(palette)):

        rgb[mask == class_id] = palette[class_id]

    return Image.fromarray(rgb)


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_patch_visualization(
    sample,
    patch_index,
    image_root
):

    folder = os.path.join(
        image_root,
        f"patch_{patch_index:03d}"
    )

    os.makedirs(
        folder,
        exist_ok=True
    )

    raw = sample["raw"]

    # raw обычно [C, Z, Y, X]
    if raw.ndim == 4:
        raw = raw[0]

    gt = sample["gt"]
    pred = sample["pred"]

    classes = sample.get(
        "classes",
        []
    )

    print(
        f"Patch {patch_index:03d}: "
        f"classes={classes}"
    )

    # --------------------------------------------------------
    # Центральный Z-срез
    # --------------------------------------------------------

    z_center = raw.shape[0] // 2

    raw_view = raw[
        z_center,
        :,
        :
    ]

    gt_view = gt[
        z_center,
        :,
        :
    ]

    pred_view = pred[
        z_center,
        :,
        :
    ]

    # --------------------------------------------------------
    # Сохраняем 3 изображения прямо в patch_XXX
    # --------------------------------------------------------

    normalize_seismic(
        raw_view
    ).save(
        os.path.join(
            folder,
            "seismic.png"
        )
    )

    colorize_mask(
        gt_view
    ).save(
        os.path.join(
            folder,
            "ground_truth.png"
        )
    )

    colorize_mask(
        pred_view
    ).save(
        os.path.join(
            folder,
            "prediction.png"
        )
    )
# ============================================================
# PRINT CONFUSION MATRIX
# ============================================================

def print_confusion_matrix(
    cm,
    n_classes
):

    print()
    print("=" * 120)
    print("CONFUSION MATRIX — PERCENTAGES")
    print("Rows = TRUE CLASS")
    print("Columns = PREDICTED CLASS")
    print("Each row is normalized to 100%")
    print("=" * 120)

    cm_percent = cm.astype(
        np.float64
    )

    row_sums = cm_percent.sum(
        axis=1,
        keepdims=True
    )

    cm_percent = np.divide(
        cm_percent,
        row_sums,
        out=np.zeros_like(cm_percent),
        where=row_sums != 0
    ) * 100.0

    print(
        f"{'True/Pred':>12}",
        end=""
    )

    for class_id in range(n_classes):

        print(
            f"{class_id:>10}",
            end=""
        )

    print()

    print("-" * 120)

    for true_class in range(n_classes):

        print(
            f"{true_class:>12}",
            end=""
        )

        for pred_class in range(n_classes):

            print(
                f"{cm_percent[true_class, pred_class]:>9.2f}%",
                end=""
            )

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

        row = cm_percent[
            true_class
        ].copy()

        # Убираем правильное предсказание
        row[true_class] = 0

        sorted_indices = np.argsort(
            row
        )[::-1]

        errors = []

        for pred_class in sorted_indices[:3]:

            if row[pred_class] > 0:

                errors.append(
                    f"{pred_class} "
                    f"({row[pred_class]:.2f}%)"
                )

        if errors:

            print(
                f"True class {true_class}: "
                + ", ".join(errors)
            )

        else:

            print(
                f"True class {true_class}: "
                "no errors"
            )

    print("=" * 100)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print()
    print("=" * 80)
    print("3D U-NET TEST")
    print("=" * 80)

    print(
        f"Checkpoint : {args.checkpoint}"
    )

    print(
        f"Test H5    : {args.h5}"
    )

    print(
        f"Config     : {args.config}"
    )

    print(
        f"Classes    : {args.n_classes}"
    )

    print(
        f"Batch size : {args.batch_size}"
    )

    print(
        f"Visualize  : {args.max_vis_samples}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # LOAD CONFIG
    # --------------------------------------------------------

    print()
    print("Loading config...")

    # load_config обычно ожидает sys.argv.
    # Поэтому временно подменяем argv.
    old_argv = sys.argv

    sys.argv = [
        "test.py",
        "--config",
        args.config
    ]

    try:

        config, _ = load_config()

    finally:

        sys.argv = old_argv

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("Creating model...")

    model = get_model(
    config["model"]
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
          weights_only=False
    )
    
    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif isinstance(
        checkpoint,
        dict
    ) and "state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model = model.to(
        device
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # HDF5 CHECK
    # --------------------------------------------------------

    print()
    print("Checking HDF5...")

    with h5py.File(
        args.h5,
        "r"
    ) as f:

        print(
            "Datasets in HDF5:"
        )

        for key in f.keys():

            print(
                f"  - {key}"
            )

        if "raw" not in f:

            raise RuntimeError(
                "Dataset 'raw' not found in HDF5."
            )

        if "label" not in f:

            raise RuntimeError(
                "Dataset 'label' not found in HDF5."
            )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print()
    print("Creating test dataset...")

    # ВАЖНО:
    # Здесь используем только raw + label.
    # Никаких random augmentations.

    dataset = StandardHDF5Dataset(
        file_path=args.h5,
        phase="test",
       slice_builder_config={
            "name": "SliceBuilder",
            "ndim": 3,
            "patch_shape": [
                16,
                128,
                128
            ],
            "stride_shape": [
                8,
                64,
                64
            ]
        },
        transformer_config={
            "label": [
                {
                    "name": "ToTensor",
                    "dtype": "long",
                    "expand_dims": False
                }
            ],
            "raw": [
                {
                    "name": "Normalize"
                },
                {
                    "name": "ToTensor",
                    "expand_dims": True
                }
            ]
        }
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda batch: batch[0]
    )

    print(
        f"Number of test patches: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORIES
    # --------------------------------------------------------

    image_root = os.path.join(
        args.output_dir,
        "visualizations"
    )

    os.makedirs(
        image_root,
        exist_ok=True
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # METRIC STORAGE
    # --------------------------------------------------------

    all_targets = []
    all_preds = []

    saved_slices = []

    covered_classes = set()

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RUNNING TEST")
    print("=" * 80)

    with torch.no_grad():


        test_h5 = h5py.File(args.h5, "r")
        label_dataset = test_h5["label"]

        for batch_idx, batch in enumerate(tqdm(
                loader,
                desc="Testing",
                total=len(loader),
                unit="batch"
            )
        ):

            # ============================================================
            # BATCH STRUCTURE
            # ============================================================

            input_tensor = batch[0]

            patch_slice = batch[1]


            # ============================================================
            # ADD CHANNEL DIMENSION
            # ============================================================

            # Dataset возвращает:
            # [B, D, H, W]
            #
            # UNet3D ожидает:
            # [B, C, D, H, W]

            if input_tensor.ndim == 4:
                input_tensor = input_tensor.unsqueeze(1)

            input_tensor = input_tensor.to(
                device,
                non_blocking=True
            )

            # ============================================================
            # GET GROUND TRUTH LABEL FROM HDF5
            # ============================================================

          

            label_np = label_dataset[patch_slice]

            target_tensor = torch.from_numpy(
                np.asarray(label_np)
            ).long()

            # Добавляем batch dimension
            #
            # [D,H,W]
            #     ↓
            # [B,D,H,W]

            target_tensor = target_tensor.unsqueeze(0).to(
                device,
                non_blocking=True
            )

        
            # ============================================================
            # MODEL PREDICTION
            # ============================================================

            output = model(
                input_tensor
            )

            prediction = torch.argmax(
                output,
                dim=1
            )

            # ============================================================
            # FLATTEN METRICS
            # ============================================================

            target_np = (
                target_tensor
                .cpu()
                .numpy()
                .reshape(-1)
            )

            pred_np = (
                prediction
                .cpu()
                .numpy()
                .reshape(-1)
            )

            all_targets.append(
                target_np
            )

            all_preds.append(
                pred_np
            )

            # ============================================================
            # VISUALIZATION SELECTION
            # ============================================================

            if len(saved_slices) < args.max_vis_samples:

                batch_size = input_tensor.shape[0]

                for sample_idx in range(batch_size):

                    gt_sample = (
                        target_tensor[
                            sample_idx
                        ]
                        .cpu()
                        .numpy()
                    )

                    pred_sample = (
                        prediction[
                            sample_idx
                        ]
                        .cpu()
                        .numpy()
                    )

                    raw_sample = (
                        input_tensor[
                            sample_idx
                        ]
                        .cpu()
                        .numpy()
                    )

                    # Убираем channel dimension
                    # [1,D,H,W] -> [D,H,W]

                    if raw_sample.ndim == 4 and raw_sample.shape[0] == 1:
                        raw_sample = raw_sample[0]

                    # ====================================================
                    # CLASSES IN THIS PATCH
                    # ====================================================

                    patch_classes = set(
                        np.unique(
                            gt_sample
                        ).tolist()
                    )

                    new_classes = (
                        patch_classes
                        - covered_classes
                    )

                    # ====================================================
                    # SAVE PATCH
                    # ====================================================

                    should_save = (
                        bool(new_classes)
                        or
                        len(saved_slices)
                        < min(
                            10,
                            args.max_vis_samples
                        )
                    )

                    if should_save:

                        saved_slices.append(
                            {
                                "raw": raw_sample,
                                "gt": gt_sample,
                                "pred": pred_sample,
                                "classes": sorted(
                                    patch_classes
                                ),
                                "slice": patch_slice
                            }
                        )

                        covered_classes.update(
                            patch_classes
                        )

                    if (
                        len(saved_slices)
                        >= args.max_vis_samples
                    ):
                        break


        test_h5.close()

    # --------------------------------------------------------
    # CONCAT METRICS
    # --------------------------------------------------------

    all_targets = np.concatenate(
        all_targets
    )

    all_preds = np.concatenate(
        all_preds
    )

    print()
    print("=" * 80)
    print("TEST FINISHED")
    print("=" * 80)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_targets,
        all_preds,
        labels=list(
            range(args.n_classes)
        )
    )

    print_confusion_matrix(
        cm,
        args.n_classes
    )

    # --------------------------------------------------------
    # CLASS METRICS
    # --------------------------------------------------------

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            all_targets,
            all_preds,
            labels=list(
                range(args.n_classes)
            ),
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou = []

    for class_id in range(
        args.n_classes
    ):

        true_positive = cm[
            class_id,
            class_id
        ]

        false_positive = (
            cm[:, class_id].sum()
            - true_positive
        )

        false_negative = (
            cm[class_id, :].sum()
            - true_positive
        )

        denominator = (
            true_positive
            + false_positive
            + false_negative
        )

        if denominator == 0:

            class_iou = 0.0

        else:

            class_iou = (
                true_positive
                / denominator
            )

        iou.append(
            class_iou
        )

    iou = np.array(
        iou
    )

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    pixel_accuracy = (
        np.mean(
            all_targets
            == all_preds
        )
    )

    class_accuracy = []

    for class_id in range(
        args.n_classes
    ):

        class_mask = (
            all_targets
            == class_id
        )

        if class_mask.sum() == 0:

            class_acc = 0.0

        else:

            class_acc = np.mean(
                all_preds[class_mask]
                == class_id
            )

        class_accuracy.append(
            class_acc
        )

    class_accuracy = np.array(
        class_accuracy
    )

    mean_class_accuracy = (
        class_accuracy.mean()
    )

    mean_iou = (
        iou.mean()
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("FINAL TEST RESULTS")
    print("=" * 100)

    print(
        f"Pixel Accuracy       : "
        f"{pixel_accuracy:.5f}"
    )

    print(
        f"Mean Class Accuracy  : "
        f"{mean_class_accuracy:.5f}"
    )

    print(
        f"Mean IoU             : "
        f"{mean_iou:.5f}"
    )

    print()
    print(
        f"{'Class':>8}"
        f"{'Accuracy':>14}"
        f"{'IoU':>14}"
        f"{'Precision':>14}"
        f"{'Recall':>14}"
        f"{'F1':>14}"
        f"{'Pixels':>14}"
    )

    print("-" * 100)

    for class_id in range(
        args.n_classes
    ):

        print(
            f"{class_id:>8}"
            f"{class_accuracy[class_id]:>13.4f}"
            f"{iou[class_id]:>14.4f}"
            f"{precision[class_id]:>14.4f}"
            f"{recall[class_id]:>14.4f}"
            f"{f1[class_id]:>14.4f}"
            f"{support[class_id]:>14}"
        )

    print("-" * 100)

    # --------------------------------------------------------
    # SAVE CONFUSION MATRIX CSV
    # --------------------------------------------------------

    cm_percent = cm.astype(
        np.float64
    )

    row_sums = cm_percent.sum(
        axis=1,
        keepdims=True
    )

    cm_percent = np.divide(
        cm_percent,
        row_sums,
        out=np.zeros_like(cm_percent),
        where=row_sums != 0
    ) * 100.0

    cm_csv = os.path.join(
        args.output_dir,
        "confusion_matrix_percent.csv"
    )

    np.savetxt(
        cm_csv,
        cm_percent,
        delimiter=",",
        fmt="%.4f"
    )

    print()
    print(
        f"Confusion matrix saved to:"
        f"\n{cm_csv}"
    )

    # --------------------------------------------------------
    # SAVE NUMERICAL RESULTS
    # --------------------------------------------------------

    results_csv = os.path.join(
        args.output_dir,
        "class_metrics.csv"
    )

    # --------------------------------------------------------
    # Средние значения по классам
    # --------------------------------------------------------

    mean_accuracy = class_accuracy.mean()
    mean_iou = iou.mean()
    mean_precision = precision.mean()
    mean_recall = recall.mean()
    mean_f1 = f1.mean()

    # --------------------------------------------------------
    # Результаты по каждому классу
    # --------------------------------------------------------

    results_df = pd.DataFrame({
        "class": np.arange(
            args.n_classes
        ),

        "accuracy": class_accuracy,

        "iou": iou,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "pixels": support
    })

    # --------------------------------------------------------
    # Добавляем Mean
    # --------------------------------------------------------

    mean_row = pd.DataFrame([
        {
            "class": "Mean",

            "accuracy": mean_accuracy,

            "iou": mean_iou,

            "precision": mean_precision,

            "recall": mean_recall,

            "f1": mean_f1,

            "pixels": ""
        }
    ])

    results_df = pd.concat(
        [
            results_df,
            mean_row
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Сохраняем CSV
    # --------------------------------------------------------

    results_df.to_csv(
        results_csv,
        index=False,
        float_format="%.6f"
    )

    print(
        f"Class metrics saved to:"
        f"\n{results_csv}"
    )

    # --------------------------------------------------------
    # SAVE VISUALIZATIONS
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        f"SAVING {len(saved_slices)} "
        f"VISUALIZATION PATCHES"
    )
    print("=" * 80)

    for idx, sample in enumerate(
        saved_slices
    ):

        save_patch_visualization(
            sample,
            idx,
            image_root
        )

    print()
    print(
        f"Images saved to:"
        f"\n{image_root}"
    )

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("VISUALIZATION CLASS COVERAGE")
    print("=" * 80)

    print(
        "Classes covered by selected patches:"
    )

    print(
        sorted(
            covered_classes
        )
    )

    missing_classes = (
        set(
            range(
                args.n_classes
            )
        )
        - covered_classes
    )

    if missing_classes:

        print(
            "WARNING: classes not covered:"
        )

        print(
            sorted(
                missing_classes
            )
        )

    else:

        print(
            "All classes are represented "
            "in selected patches."
        )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()