import argparse
import gc
import os
import subprocess
import sys

import shutil

import h5py
import numpy as np
import yaml


def compute_class_weights(labels, n_classes):
    class_counts = np.bincount(labels.flatten(), minlength=n_classes)
    total = np.sum(class_counts)
    weights = total / (class_counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    return weights.tolist()


def train_amplitude_range(paths, low_pct=0.5, high_pct=99.5, max_samples=5_000_000, seed=0):
    """Global amplitude range estimated on the TRAIN volumes only.

    Percentiles instead of min/max: a single strong reflector must not compress the rest of
    the signal. The same fixed range is used for train, val and test, so a voxel gets the
    same normalized value in every patch that covers it.
    """
    rng = np.random.default_rng(seed)
    sizes = []
    for path in paths:
        with h5py.File(path, "r") as f:
            sizes.append(int(np.prod(f["raw"].shape)))
    total = sum(sizes)
    samples = []
    for path, size in zip(paths, sizes):
        with h5py.File(path, "r") as f:
            raw = f["raw"][:].reshape(-1)
        take = min(size, max(1, int(max_samples * size / total)))
        idx = rng.choice(size, size=take, replace=False) if take < size else slice(None)
        samples.append(np.asarray(raw[idx], dtype=np.float64))
        del raw
    values = np.concatenate(samples)
    low, high = np.percentile(values, [low_pct, high_pct])
    if not high > low:
        raise ValueError(f"Degenerate amplitude range: [{low}, {high}]")
    return float(low), float(high)


def parse_args():
    parser = argparse.ArgumentParser(description="3D U-Net training with dynamic split")
    parser.add_argument(
        "--data_h5",
        type=str,
        default=None,
        help="Один куб all_data.h5 (обязателен для всех split, кроме ready)",
    )
    parser.add_argument("--train_h5", type=str, default=None, help="Готовый train.h5, только для --split ready")
    parser.add_argument("--val_h5", type=str, default=None, help="Готовый val.h5, только для --split ready")
    parser.add_argument("--per_val", type=float, default=0.2)
    parser.add_argument("--per_test", type=float, default=0.15)
    parser.add_argument("--arch", type=str, default="UNet3D", choices=["UNet3D", "ResidualUNet3D", "ResidualUNetSE3D"])
    parser.add_argument("--split", type=str, default="spatial", choices=["spatial", "random", "e5", "reverse", "ready"])
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--f_maps", type=int, default=32)
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--early_stop",
        type=int,
        default=10,
        help="Остановка после N эпох без улучшения val-метрики; 0 отключает",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="global",
        choices=["global", "patch"],
        help="global: фиксированный диапазон амплитуд по перцентилям train (по умолчанию); "
        "patch: старый min-max по каждому патчу (для воспроизведения прежних результатов)",
    )
    parser.add_argument("--norm_low_pct", type=float, default=0.5)
    parser.add_argument("--norm_high_pct", type=float, default=99.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patch_shape", type=int, nargs=3, default=[16, 128, 128])
    parser.add_argument("--stride_shape", type=int, nargs=3, default=[8, 64, 64])
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--absolute_depth", action="store_true")
    parser.add_argument("--rot_angle", type=float, default=10.0)
    parser.add_argument(
        "--loss",
        type=str,
        default="DiceCrossEntropyLoss",
        choices=["GeneralizedDiceLoss", "CrossEntropyLoss", "DiceCrossEntropyLoss"],
    )
    parser.add_argument("--class_weights", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--config_dir", type=str, default="my_configs")
    parser.add_argument("--temp_dir", type=str, default="temp_splits")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def _tensor(expand_dims, dtype=None):
    item = {"name": "ToTensor", "expand_dims": expand_dims}
    if dtype is not None:
        item["dtype"] = dtype
    return item


def _normalize_transform(amp_range):
    if amp_range is None:
        return {"name": "Normalize"}  # per-patch min-max (legacy)
    return {"name": "Normalize", "min_value": amp_range[0], "max_value": amp_range[1]}


def config_path_for(args):
    return os.path.join(args.config_dir, f"generated_config_{args.split}.yaml")


def generate_config(args, train_paths, val_path, class_weights=None, amp_range=None):
    # Normalization is the same transform in both modes. With --absolute_depth the dataset
    # appends the depth channel after the transforms, so it is not normalized or rotated.
    raw_train = [_normalize_transform(amp_range)]
    raw_val = [_normalize_transform(amp_range), _tensor(True)]
    label_train = []
    label_val = [_tensor(False, "long")]

    if args.aug:
        raw_train.extend([
            {"name": "RandomRotate", "angle_spectrum": args.rot_angle, "order": 1, "axes": [[1, 2]]},
            {"name": "RandomFlip"},
        ])
        label_train.extend([
            {"name": "RandomRotate", "angle_spectrum": args.rot_angle, "order": 0, "axes": [[1, 2]]},
            {"name": "RandomFlip"},
        ])

    raw_train.append(_tensor(True))
    label_train.append(_tensor(False, "long"))

    config = {
        "manual_seed": args.seed,
        "loaders": {
            "dataset": "LazyHDF5DatasetWithDepth" if args.absolute_depth else "LazyHDF5Dataset",
            "train": {
                "file_paths": train_paths,
                "slice_builder": {
                    "ndim": 3,
                    "name": "SliceBuilder",
                    "patch_shape": args.patch_shape,
                    "stride_shape": args.stride_shape,
                },
                "transformer": {"raw": raw_train, "label": label_train},
            },
            "val": {
                "file_paths": [val_path],
                "slice_builder": {
                    "ndim": 3,
                    "name": "SliceBuilder",
                    "patch_shape": args.patch_shape,
                    "stride_shape": args.stride_shape,
                },
                "transformer": {"raw": raw_val, "label": label_val},
            },
            "num_workers": args.num_workers,
            "batch_size": args.batch_size,
        },
        "model": {
            "name": args.arch,
            "in_channels": 2 if args.absolute_depth else 1,
            "out_channels": args.n_classes,
            "f_maps": args.f_maps,
            "layer_order": "gcr",
            "num_groups": 8,
            "is_segmentation": True,
            "final_sigmoid": False,
        },
        "loss": {
            "name": args.loss,
            "dice_weight": 0.5,
            "ce_weight": 0.5,
            "normalization": "softmax",
        },
        "optimizer": {
            "name": "Adam",
            "learning_rate": args.lr,
            "weight_decay": 0.00001,
        },
        "trainer": {
            "checkpoint_dir": args.checkpoint_dir,
            "max_num_iterations": None,
            "max_num_epochs": args.n_epoch,
            "resume": args.resume,
            "patience": args.early_stop,
        },
        "eval_metric": {
            "name": "SegmentationMetrics",
            "n_classes": args.n_classes,
        },
        "logging": {
            "log_dir": args.log_dir,
            "tensorboard": True,
        },
        # Not used by pytorch3dunet; kept so that every checkpoint has a full record of the run
        "experiment": {
            "split": args.split,
            "norm": args.norm,
            "amplitude_range": list(amp_range) if amp_range is not None else None,
            "absolute_depth": bool(args.absolute_depth),
            "aug": bool(args.aug),
        },
    }

    if args.loss in ("CrossEntropyLoss", "DiceCrossEntropyLoss") and args.class_weights and class_weights is not None:
        config["loss"]["weight"] = class_weights

    os.makedirs(args.config_dir, exist_ok=True)
    config_path = config_path_for(args)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    # A copy next to the checkpoints: the test script can always find the config of a model
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    shutil.copyfile(config_path, os.path.join(args.checkpoint_dir, "config.yaml"))
    return config_path


def save_volume(path, raw, label):
    with h5py.File(path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("label", data=label, compression="gzip")


def class_weights_from_files(paths, n_classes):
    parts = []
    for path in paths:
        with h5py.File(path, "r") as f:
            parts.append(f["label"][:])
    labels = np.concatenate(parts, axis=0)
    weights = compute_class_weights(labels, n_classes)
    del labels, parts
    gc.collect()
    return weights


def launch(args, train_paths, val_path, class_weights=None):
    amp_range = None
    if args.norm == "global":
        amp_range = train_amplitude_range(train_paths, args.norm_low_pct, args.norm_high_pct, seed=args.seed)
        print(f"Диапазон амплитуд по train (p{args.norm_low_pct}..p{args.norm_high_pct}): {amp_range}")
    config_path = generate_config(args, train_paths, val_path, class_weights, amp_range)
    print(f"Конфиг сохранён: {config_path}")
    cmd = [sys.executable, "-m", "pytorch3dunet.train", "--config", config_path]
    if args.resume:
        print(f"Возобновление с чекпоинта: {args.resume}")
    print("Запуск:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_ready(args):
    if not args.train_h5 or not args.val_h5:
        raise ValueError("Для --split ready --train_h5 и --val_h5")
    for path in (args.train_h5, args.val_h5):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    print("READY SPLIT")
    print(f"Train: {args.train_h5}")
    print(f"Val:   {args.val_h5}")
    print("Индексы [0:619] / [809:951] не применяются.")

    weights = None
    if args.class_weights and args.loss in ("CrossEntropyLoss", "DiceCrossEntropyLoss"):
        print("Вычисление весов классов...")
        weights = class_weights_from_files([args.train_h5], args.n_classes)
        print("Веса:", weights)

    launch(args, [args.train_h5], args.val_h5, weights)


def make_indices(split, n, per_val, per_test, seed):
    test_size = int(n * per_test)
    val_size = int(n * per_val)
    train_size = n - test_size - val_size

    if split == "spatial":
        train = [np.arange(0, train_size)]
        val = np.arange(train_size, train_size + val_size)
        test = np.arange(train_size + val_size, n)
        print("SPATIAL SPLIT")
        print(f"Train: {len(train[0])} [0:{train_size}]")
        print(f"Val:   {len(val)} [{train_size}:{train_size + val_size}]")
        print(f"Test:  {len(test)} [{train_size + val_size}:{n}]")
        return train, val, test

    if split == "e5":
        if n < 951:
            raise ValueError(f"E5 рассчитан на куб от 951 срезов, сейчас {n}")
        train = [np.arange(0, 310), np.arange(452, 761)]
        test = np.arange(310, 452)
        val = np.arange(761, 951)
        print("E5 SPATIAL SPLIT")
        print(f"Train 1: {len(train[0])} [0:310]")
        print(f"Test:    {len(test)} [310:452]")
        print(f"Train 2: {len(train[1])} [452:761]")
        print(f"Val:     {len(val)} [761:951]")
        print(f"Total train: {len(train[0]) + len(train[1])}")
        return train, val, test

    if split == "reverse":
        if n != 951:
            raise ValueError(
                f"Reverse split рассчитан на 951 срез, сейчас {n}. "
                "Не запускайте его на объёме другой длины без пересмотра диапазонов."
            )
        test = np.arange(0, 142)
        val = np.arange(142, 332)
        train = [np.arange(332, 951)]
        print("REVERSE SPATIAL SPLIT")
        print(f"Test:  {len(test)} [0:142]")
        print(f"Val:   {len(val)} [142:332]")
        print(f"Train: {len(train[0])} [332:951]")
        return train, val, test

    if split == "random":
        rng = np.random.default_rng(seed)
        indices = np.arange(n)
        rng.shuffle(indices)
        train_idx = indices[:train_size]
        val = indices[train_size:train_size + val_size]
        test = indices[train_size + val_size:]
        print("RANDOM SPLIT")
        print(f"Train: {len(train_idx)}")
        print(f"Val:   {len(val)}")
        print(f"Test:  {len(test)}")
        print("Train ∩ Val:", len(set(train_idx.tolist()) & set(val.tolist())))
        print("Train ∩ Test:", len(set(train_idx.tolist()) & set(test.tolist())))
        print("Val ∩ Test:", len(set(val.tolist()) & set(test.tolist())))
        return [train_idx], val, test

    raise ValueError(f"Неизвестный split: {split}")


def run_from_cube(args):
    if not args.data_h5:
        raise ValueError("Укажите --data_h5 (путь к all_data.h5)")
    if not os.path.isfile(args.data_h5):
        raise FileNotFoundError(args.data_h5)
    print(f"Загрузка {args.data_h5} ...")
    with h5py.File(args.data_h5, "r") as f:
        raw = f["raw"][:]
        label = f["label"][:]
    n = raw.shape[0]
    print(f"Всего срезов: {n}")

    train_parts, val_idx, test_idx = make_indices(args.split, n, args.per_val, args.per_test, args.seed)

    out_dir = os.path.join(args.temp_dir, args.split)
    os.makedirs(out_dir, exist_ok=True)

    train_paths = []
    for i, idx in enumerate(train_parts, start=1):
        name = "train.h5" if len(train_parts) == 1 else f"train_{i}.h5"
        path = os.path.join(out_dir, name)
        save_volume(path, raw[idx], label[idx])
        train_paths.append(path)

    val_path = os.path.join(out_dir, "val.h5")
    test_path = os.path.join(out_dir, "test.h5")
    save_volume(val_path, raw[val_idx], label[val_idx])
    save_volume(test_path, raw[test_idx], label[test_idx])
    print(f"Test сохранён и в обучении не используется: {test_path}")

    del raw, label
    gc.collect()

    weights = None
    if args.class_weights and args.loss in ("CrossEntropyLoss", "DiceCrossEntropyLoss"):
        print("Вычисление весов классов...")
        weights = class_weights_from_files(train_paths, args.n_classes)
        print("Веса:", weights)

    launch(args, train_paths, val_path, weights)


def main():
    args = parse_args()
    for folder in (args.temp_dir, args.checkpoint_dir, args.log_dir, args.config_dir):
        os.makedirs(folder, exist_ok=True)

    if args.split == "ready":
        run_ready(args)
    else:
        run_from_cube(args)


if __name__ == "__main__":
    main()