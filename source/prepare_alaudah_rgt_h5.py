
from pathlib import Path
import json

import h5py
import numpy as np


# =========================================================
# PATHS
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

MY_DATA = (
    PROJECT_DIR
    / "pytorch-3dunet-master"
    / "my_data"
)

BENCHMARK_DIR = MY_DATA / "benchmark"
RGT_DIR = SCRIPT_DIR / "rgt_output" / "alaudah"
OUTPUT_DIR = MY_DATA / "benchmark_rgt"

META_FILE = RGT_DIR / "rgt_metadata.json"

SPLIT_SHAPES = {
    "train": (631, 255, 401),
    "val": (70, 255, 401),
    "test1": (701, 255, 200),
    "test2": (200, 255, 601),
}

CROSSLINE_CHUNK = 16


# =========================================================
# VALIDATE RGT
# =========================================================

def validate_rgt_metadata():

    if not META_FILE.exists():
        raise FileNotFoundError(
            f"Не найдены метаданные RGT: {META_FILE}"
        )

    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    fill_distance = metadata.get(
        "max_horizon_fill_distance"
    )

    if fill_distance is None:
        raise RuntimeError(
            "В metadata нет max_horizon_fill_distance. "
            "Пересоздай RGT актуальным build_alaudah_rgt.py."
        )

    print("RGT fill distance:", fill_distance)

    if float(fill_distance) > 2.0001:
        raise RuntimeError(
            "Обнаружен старый RGT, построенный с радиусом "
            f"заполнения {fill_distance}.\n"
            "Измени MAX_HORIZON_FILL_DISTANCE на 2.0, "
            "перезапусти build_alaudah_rgt.py и повтори конвертацию."
        )


# =========================================================
# CONVERT ONE SPLIT
# =========================================================

def convert_split(name, expected_shape):

    source_path = BENCHMARK_DIR / f"{name}.h5"
    rgt_path = RGT_DIR / f"rgt_{name}.npy"
    output_path = OUTPUT_DIR / f"{name}.h5"

    for path in (source_path, rgt_path):
        if not path.exists():
            raise FileNotFoundError(path)

    print()
    print("=" * 65)
    print(f"CONVERT {name.upper()}")
    print("=" * 65)

    rgt = np.load(
        rgt_path,
        mmap_mode="r",
    )

    if tuple(rgt.shape) != expected_shape:
        raise ValueError(
            f"{name}: RGT shape {rgt.shape}, "
            f"ожидалось {expected_shape}"
        )

    with h5py.File(source_path, "r") as src:

        raw_src = src["raw"]
        label_src = src["label"]

        if tuple(raw_src.shape) != expected_shape:
            raise ValueError(
                f"{name}: raw shape {raw_src.shape}"
            )

        if tuple(label_src.shape) != expected_shape:
            raise ValueError(
                f"{name}: label shape {label_src.shape}"
            )

        c_size, d_size, i_size = expected_shape

        chunks_raw = (
            1,
            min(CROSSLINE_CHUNK, c_size),
            min(128, d_size),
            min(128, i_size),
        )

        chunks_label = (
            min(CROSSLINE_CHUNK, c_size),
            min(128, d_size),
            min(128, i_size),
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        with h5py.File(output_path, "w") as dst:

            raw_out = dst.create_dataset(
                "raw",
                shape=(2, c_size, d_size, i_size),
                dtype=np.float32,
                chunks=chunks_raw,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            label_out = dst.create_dataset(
                "label",
                shape=expected_shape,
                dtype=label_src.dtype,
                chunks=chunks_label,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )

            # Метаданные
            dst.attrs["axis_order"] = (
                "channel, crossline, depth, inline"
            )

            dst.attrs["channel_names"] = (
                "seismic,rgt"
            )

            dst.attrs["n_classes"] = 6
            dst.attrs["rgt_range"] = "0..1"

            # Копируем блоками по crossline,
            # чтобы не загружать весь объём в RAM.
            for c0 in range(
                0,
                c_size,
                CROSSLINE_CHUNK,
            ):

                c1 = min(
                    c0 + CROSSLINE_CHUNK,
                    c_size,
                )

                seismic_block = np.asarray(
                    raw_src[c0:c1, :, :],
                    dtype=np.float32,
                )

                rgt_block = np.asarray(
                    rgt[c0:c1, :, :],
                    dtype=np.float32,
                )

                label_block = label_src[
                    c0:c1, :, :
                ]

                if not np.isfinite(rgt_block).all():
                    raise ValueError(
                        f"{name}: RGT содержит NaN/Inf"
                    )

                if (
                    rgt_block.min() < -1e-5
                    or rgt_block.max() > 1.00001
                ):
                    raise ValueError(
                        f"{name}: RGT вне диапазона 0..1: "
                        f"{rgt_block.min()}..{rgt_block.max()}"
                    )

                # Первый канал — исходная сейсмика.
                raw_out[0, c0:c1, :, :] = seismic_block

                # Второй канал — RGT.
                raw_out[1, c0:c1, :, :] = rgt_block

                # Разметка не меняется.
                label_out[c0:c1, :, :] = label_block

                print(
                    f"  записано crossline: {c0}:{c1}/{c_size}",
                    end="\r",
                    flush=True,
                )

            dst.attrs["source_h5"] = str(source_path)
            dst.attrs["source_rgt"] = str(rgt_path)

    print()
    print("Сохранено:", output_path)

    with h5py.File(output_path, "r") as check:

        print("raw:", check["raw"].shape)
        print("label:", check["label"].shape)

        print(
            "Размер файла:",
            f"{output_path.stat().st_size / (1024 ** 2):.1f} MB"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 65)
    print("PREPARE ALAUDAH HDF5 WITH RGT")
    print("=" * 65)

    validate_rgt_metadata()

    for name, shape in SPLIT_SHAPES.items():
        convert_split(name, shape)

    metadata = {
        "axis_order_raw": "channel, crossline, depth, inline",
        "channels": [
            "seismic",
            "rgt",
        ],
        "label_order": "crossline, depth, inline",
        "n_classes": 6,
        "source": str(BENCHMARK_DIR),
        "rgt_source": str(RGT_DIR),
        "output": str(OUTPUT_DIR),
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print()
    print("=" * 65)
    print("CONVERSION COMPLETE")
    print("=" * 65)
    print("Результаты:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
