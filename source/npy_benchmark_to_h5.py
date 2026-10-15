"""
npy бенчмарка Alaudah -> h5 (raw / label).

Официальная раскладка после unzip data.zip:
    data/train/train_seismic.npy
    data/train/train_labels.npy
    data/test_once/test1_seismic.npy
    data/test_once/test1_labels.npy
    data/test_once/test2_seismic.npy
    data/test_once/test2_labels.npy


Запуск:
    python npy_benchmark_to_h5.py --data_dir ПАПКА_С_NPY --out_dir ПАПКА_H5
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

SPLITS = ("train", "test1", "test2")


def find_npy(data_dir: Path, split: str, kind: str) -> Path:
    """kind: seismic или labels."""
    names = [
        f"{split}_{kind}.npy",
        f"F3_{split}_{kind}.npy",
        f"f3_{split}_{kind}.npy",
    ]
    folders = [
        data_dir,
        data_dir / split,
        data_dir / "test_once",
        data_dir / "data" / split,
        data_dir / "data" / "test_once",
        data_dir / "data" / "train",
    ]
    if split == "train":
        folders.append(data_dir / "train")
        folders.append(data_dir / "data" / "train")

    for folder in folders:
        for name in names:
            path = folder / name
            if path.is_file():
                return path
    raise FileNotFoundError(
        f"Не нашёл {split}_{kind}.npy в {data_dir}. "
        f"Положи туда train/test1/test2 seismic и labels."
    )


def load_pair(data_dir: Path, split: str):
    seismic = np.load(find_npy(data_dir, split, "seismic"))
    labels = np.load(find_npy(data_dir, split, "labels"))
    if seismic.shape != labels.shape:
        raise ValueError(f"{split}: формы не совпали {seismic.shape} vs {labels.shape}")
    if seismic.ndim != 3:
        raise ValueError(f"{split}: нужен 3D, пришло {seismic.shape}")

    # (inline, crossline, depth) -> (crossline, depth, inline)
    seismic = np.transpose(seismic, (1, 2, 0)).astype(np.float32, copy=False)
    labels = np.transpose(labels, (1, 2, 0))

    uniq = np.unique(labels)
    if int(uniq.min()) == 1 and int(uniq.max()) == 6:
        print("  метки были 1..6, сдвигаю в 0..5")
        labels = labels - 1
        uniq = np.unique(labels)

    return seismic, labels.astype(np.int64, copy=False), uniq


def save_h5(path: Path, seismic: np.ndarray, labels: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("raw", data=seismic, compression="gzip", compression_opts=4)
        f.create_dataset("label", data=labels, compression="gzip", compression_opts=4)
        f.attrs["axis_order"] = "crossline, depth, inline"
        f.attrs["n_classes"] = 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--val_crosslines", type=int, default=70)
    args = parser.parse_args()

    for split in SPLITS:
        print(f"\n{split}")
        print(" ", find_npy(args.data_dir, split, "seismic"))
        print(" ", find_npy(args.data_dir, split, "labels"))
        seismic, labels, uniq = load_pair(args.data_dir, split)
        print(f"  shape (crossline, depth, inline) = {tuple(seismic.shape)}")
        print(
            f"  seismic min/mean/max = {seismic.min():.4f} / {seismic.mean():.6f} / {seismic.max():.4f}"
        )
        print(f"  labels = {uniq.tolist()}")

        if split == "train" and args.val_crosslines > 0:
            n = args.val_crosslines
            if n >= seismic.shape[0]:
                raise ValueError("--val_crosslines больше длины train")
            save_h5(args.out_dir / "train.h5", seismic[:-n], labels[:-n])
            save_h5(args.out_dir / "val.h5", seismic[-n:], labels[-n:])
            print(f"  train.h5 = crossline [0:{seismic.shape[0] - n}]")
            print(f"  val.h5   = crossline [{seismic.shape[0] - n}:{seismic.shape[0]}]")
        else:
            name = "train.h5" if split == "train" else f"{split}.h5"
            save_h5(args.out_dir / name, seismic, labels)
            print(f"  записал {name}")

    print("\nДальше обучение только на train.h5 и val.h5.")
    print("test1.h5 и test2.h5 не открывать до финального теста.")


if __name__ == "__main__":
    main()
