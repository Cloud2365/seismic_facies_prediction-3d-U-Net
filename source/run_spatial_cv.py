"""
Одна модель, четыре проверки вдоль куба.

Это не k-fold и не случайное перемешивание. Обучение одно.
Официальный результат 0.641 этим прогоном не заменяется:
здесь train короче, чтобы между ним и проверками остался зазор.

  train      [0:560]
  зазор      [560:576]   16 срезов, ни в обучение, ни в тест
  val        [576:619]   только выбор чекпоинта
  блоки теста, один и тот же чекпоинт:
      [619:714]
      [714:809]
      [809:880]
      [880:951]

Запуск из папки source, рядом с run_3d_training.py:

  python run_spatial_cv.py --data path\\to\\all_data.h5 --batch_size 4

Нормировка по патчу, без аугментации, без весов, без канала глубины.
--early_stop 0, как у официального прогона.
"""

import argparse
import csv
import os
import subprocess
import sys

import h5py
import numpy as np


BLOCKS = (
    ("619_714", 619, 714),
    ("714_809", 714, 809),
    ("809_880", 809, 880),
    ("880_951", 880, 951),
)


def save_volume(path, raw, label):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("label", data=label, compression="gzip")
    print(f"{path}  срезы, форма {raw.shape}")


def prepare(data_h5, out_dir):
    with h5py.File(data_h5, "r") as f:
        n = f["raw"].shape[0]
        if n < 951:
            raise SystemExit(f"Нужен куб от 951 среза, сейчас {n}")
        raw = f["raw"]
        label = f["label"]
        save_volume(os.path.join(out_dir, "train.h5"), raw[0:560], label[0:560])
        save_volume(os.path.join(out_dir, "val.h5"), raw[576:619], label[576:619])
        for name, start, stop in BLOCKS:
            save_volume(
                os.path.join(out_dir, f"test_{name}.h5"),
                raw[start:stop],
                label[start:stop],
            )
    print("Зазор [560:576] не записан.")


def train(args, out_dir):
    cmd = [
        sys.executable,
        os.path.join(args.root, "run_3d_training.py"),
        "--data_h5",
        args.data,
        "--split",
        "ready",
        "--train_h5",
        os.path.join(out_dir, "train.h5"),
        "--val_h5",
        os.path.join(out_dir, "val.h5"),
        "--norm",
        "patch",
        "--early_stop",
        "0",
        "--n_epoch",
        str(args.n_epoch),
        "--batch_size",
        str(args.batch_size),
        "--checkpoint_dir",
        args.checkpoint_dir,
        "--log_dir",
        args.log_dir,
        "--config_dir",
        args.config_dir,
        "--temp_dir",
        os.path.join(args.temp_dir, "unused"),
    ]
    print("Запуск:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def test_blocks(args):
    config = os.path.join(args.checkpoint_dir, "config.yaml")
    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f"Нет чекпоинта: {args.checkpoint}")
    if not os.path.isfile(config):
        raise SystemExit(f"Нет конфига обучения: {config}")
    out_dir = os.path.join(args.temp_dir, "cv")
    for name, _, _ in BLOCKS:
        h5_path = os.path.join(out_dir, f"test_{name}.h5")
        output_dir = os.path.join(args.test_dir, name)
        cmd = [
            sys.executable,
            os.path.join(args.root, "run_3d_test.py"),
            "--checkpoint",
            args.checkpoint,
            "--h5",
            h5_path,
            "--config",
            config,
            "--output_dir",
            output_dir,
            "--batch_size",
            str(args.test_batch_size),
            "--max_vis_samples",
            "0",
        ]
        print("Запуск:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def read_block(output_dir):
    metrics_path = os.path.join(output_dir, "class_metrics.csv")
    cm_path = os.path.join(output_dir, "confusion_matrix_percent.csv")
    with open(metrics_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    mean_iou = None
    recall6 = None
    for row in rows:
        if row["class"] == "Mean":
            mean_iou = float(row["iou"])
        if row["class"] == "6":
            recall6 = float(row["recall"])
    cm = np.loadtxt(cm_path, delimiter=",")
    # В файле уже проценты. Строка 6, столбец 5: настоящий 6 назван 5.
    as_five = float(cm[6, 5])
    return mean_iou, recall6, as_five


def report(test_dir):
    print()
    print(f"{'срезы':<16}{'mIoU':>8}{'класс6 узнан':>16}{'класс6 назван 5, %':>22}")
    for name, start, stop in BLOCKS:
        mean_iou, recall6, as_five = read_block(os.path.join(test_dir, name))
        print(
            f"{start}:{stop}".ljust(16)
            + f"{mean_iou:8.4f}"
            + f"{recall6:16.4f}"
            + f"{as_five:22.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Одна модель, четыре блока вдоль куба")
    parser.add_argument("--data", required=True, help="all_data.h5")
    parser.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--temp_dir", default="temp_splits")
    parser.add_argument("--checkpoint_dir", default=os.path.join("checkpoints", "spatial_cv"))
    parser.add_argument("--log_dir", default=os.path.join("logs", "spatial_cv"))
    parser.add_argument("--config_dir", default=os.path.join("my_configs", "spatial_cv"))
    parser.add_argument("--test_dir", default=os.path.join("test_logs", "spatial_cv"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="По умолчанию checkpoints/spatial_cv/best_checkpoint.pytorch",
    )
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.checkpoint_dir, "best_checkpoint.pytorch")

    out_dir = os.path.join(args.temp_dir, "cv")
    if not args.skip_prepare:
        prepare(args.data, out_dir)
    if not args.skip_train:
        train(args, out_dir)
    if not args.skip_test:
        test_blocks(args)
        report(args.test_dir)


if __name__ == "__main__":
    main()
