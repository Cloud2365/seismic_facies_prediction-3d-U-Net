import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import hilbert, find_peaks
from scipy.ndimage import median_filter


# Полный объём: crossline, depth, inline
FULL_SHAPE = (901, 255, 601)

# Координаты частей внутри полного объёма.
# Формат: crossline_start, crossline_end, inline_start, inline_end
SPLITS = {
    "train": (0, 631, 200, 601),
    "val":   (631, 701, 200, 601),
    "test1": (0, 701, 0, 200),
    "test2": (701, 901, 0, 601),
}


def list_datasets(h5_file):
    """Возвращает все датасеты внутри HDF5."""

    result = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            result.append((name, obj))

    h5_file.visititems(visitor)
    return result


def compatible_layout(shape, expected_shape):
    """
    Определяет, как извлечь 3D-сейсмику из датасета.

    Поддерживает:
      (C, D, I)
      (channels, C, D, I)
      (C, D, I, channels)
      дополнительные единичные измерения.
    """

    shape = tuple(shape)

    if shape == expected_shape:
        return "3d"

    if len(shape) == 4 and shape[1:] == expected_shape:
        return "channel_first"

    if len(shape) == 4 and shape[:3] == expected_shape:
        return "channel_last"

    squeezed = tuple(x for x in shape if x != 1)

    if squeezed == expected_shape:
        return "squeeze"

    return None


def choose_dataset(h5_file, expected_shape, dataset_name=None):
    """Находит датасет с сейсмикой."""

    datasets = list_datasets(h5_file)

    if dataset_name is not None:
        if dataset_name not in h5_file:
            available = [name for name, _ in datasets]
            raise KeyError(
                f"Датасет '{dataset_name}' не найден. "
                f"Доступные датасеты: {available}"
            )

        ds = h5_file[dataset_name]

        if compatible_layout(ds.shape, expected_shape) is None:
            raise ValueError(
                f"Датасет {dataset_name} имеет форму {ds.shape}, "
                f"ожидалась совместимая с {expected_shape}."
            )

        return dataset_name, ds

    priority = {
        "raw": 0,
        "seismic": 1,
        "data": 2,
        "input": 3,
        "volume": 4,
    }

    candidates = []

    for name, ds in datasets:

        basename = name.split("/")[-1].lower()

        if any(word in basename for word in ("label", "mask", "target")):
            continue

        layout = compatible_layout(ds.shape, expected_shape)

        if layout is None:
            continue

        score = priority.get(basename, 10)
        candidates.append((score, name, ds))

    if not candidates:
        available = [
            f"{name}: {ds.shape}"
            for name, ds in datasets
        ]

        raise RuntimeError(
            f"Не найден подходящий датасет для формы {expected_shape}.\n"
            f"Найденные датасеты:\n" + "\n".join(available)
        )

    candidates.sort(key=lambda x: x[0])

    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        raise RuntimeError(
            "Найдено несколько подходящих датасетов. "
            "Укажи нужный через --dataset. Кандидаты: "
            + ", ".join(x[1] for x in candidates)
        )

    return candidates[0][1], candidates[0][2]


def read_seismic_split(path, expected_shape, dataset_name=None, channel=0):
    """Читает только сейсмику из одного HDF5-файла."""

    with h5py.File(path, "r") as f:

        name, ds = choose_dataset(
            f,
            expected_shape,
            dataset_name
        )

        layout = compatible_layout(ds.shape, expected_shape)

        print(
            f"  Dataset: {name}, shape={ds.shape}, "
            f"layout={layout}"
        )

        if layout == "3d":
            data = ds[...]

        elif layout == "channel_first":
            if not 0 <= channel < ds.shape[0]:
                raise ValueError(
                    f"Канал {channel} отсутствует в {ds.shape}"
                )
            data = ds[channel, ...]

        elif layout == "channel_last":
            if not 0 <= channel < ds.shape[-1]:
                raise ValueError(
                    f"Канал {channel} отсутствует в {ds.shape}"
                )
            data = ds[..., channel]

        else:
            data = np.squeeze(ds[...])

    data = np.asarray(data, dtype=np.float32)

    if data.shape != expected_shape:
        raise ValueError(
            f"После чтения получена форма {data.shape}, "
            f"ожидалась {expected_shape}."
        )

    return data


def assemble_full_seismic(benchmark_dir, output_path, dataset_name=None, channel=0):
    """
    Собирает полный объём из train, val, test1, test2.

    Метки фаций не читаются.
    """

    benchmark_dir = Path(benchmark_dir)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_files = {
        "train": "train.h5",
        "val": "val.h5",
        "test1": "test1.h5",
        "test2": "test2.h5",
    }

    for split, filename in required_files.items():
        path = benchmark_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Не найден файл: {path}"
            )

    print("\nСоздаём полный сейсмический объём:", FULL_SHAPE)

    seismic = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=FULL_SHAPE
    )

    seismic[:] = np.nan

    for split, coords in SPLITS.items():

        filename = required_files[split]
        path = benchmark_dir / filename

        c0, c1, i0, i1 = coords

        expected_shape = (
            c1 - c0,
            FULL_SHAPE[1],
            i1 - i0
        )

        print(f"\nЧитаем {filename}")
        print("Ожидаемая форма:", expected_shape)

        data = read_seismic_split(
            path,
            expected_shape,
            dataset_name=dataset_name,
            channel=channel
        )

        seismic[c0:c1, :, i0:i1] = data

        print(
            f"Добавлено: crossline {c0}:{c1}, "
            f"inline {i0}:{i1}"
        )

        del data

    seismic.flush()

    finite_fraction = 0.0

    # Проверяем покрытие небольшими блоками.
    finite_count = 0
    total_count = 0

    for c in range(0, FULL_SHAPE[0], 20):
        block = seismic[c:c + 20]
        finite_count += np.isfinite(block).sum()
        total_count += block.size

    finite_fraction = finite_count / total_count

    print("\nСборка завершена.")
    print("Полная форма:", seismic.shape)
    print("Доля конечных значений:", round(finite_fraction * 100, 3), "%")
    print("Сохранено:", output_path)

    if finite_fraction == 0:
        raise RuntimeError("Полный объём содержит только NaN.")

    return output_path


def choose_seed_depths(seismic, count=8, min_distance=12):
    """
    Выбирает характерные глубины отражающих событий.
    Постепенно ослабляет ограничения, если пиков недостаточно.
    """

    crosslines = np.linspace(
        0,
        seismic.shape[0] - 1,
        min(100, seismic.shape[0]),
        dtype=int
    )

    inlines = np.linspace(
        0,
        seismic.shape[2] - 1,
        min(100, seismic.shape[2]),
        dtype=int
    )

    profiles = []

    for c in crosslines:
        block = np.asarray(
            seismic[c][:, inlines],
            dtype=np.float32
        )

        valid = np.isfinite(block).all(axis=0)
        block = block[:, valid]

        if block.shape[1] == 0:
            continue

        envelope = np.abs(
            hilbert(block, axis=0)
        )

        profiles.append(
            np.mean(envelope, axis=1)
        )

    if not profiles:
        raise RuntimeError("Не удалось получить профили сейсмики.")

    mean_envelope = np.mean(profiles, axis=0)

    # Сглаживаем общий профиль.
    mean_envelope = gaussian_filter1d(
        mean_envelope,
        sigma=1.5
    )

    # Не используем крайние глубины.
    margin = 12
    mean_envelope[:margin] = 0
    mean_envelope[-margin:] = 0

    std = np.std(mean_envelope)

    # Постепенно ослабляем порог выраженности и минимальное расстояние.
    attempts = [
        (0.08, min_distance),
        (0.03, min_distance),
        (0.01, max(4, min_distance // 2)),
        (0.00, max(4, min_distance // 2)),
        (0.00, 2),
        (0.00, 1),
    ]

    best_peaks = np.array([], dtype=int)

    for prominence_fraction, distance in attempts:

        peaks, _ = find_peaks(
            mean_envelope,
            distance=distance,
            prominence=std * prominence_fraction
        )

        print(
            f"Проверка: prominence={prominence_fraction}, "
            f"distance={distance}, найдено={len(peaks)}"
        )

        if len(peaks) > len(best_peaks):
            best_peaks = peaks

        if len(peaks) >= count:
            best_peaks = peaks
            break

    if len(best_peaks) < 2:
        raise RuntimeError(
            "Найдено меньше двух подходящих пиков. "
            "Нужно проверить средний профиль огибающей."
        )

    if len(best_peaks) < count:
        print(
            f"ВНИМАНИЕ: запрашивали {count} горизонтов, "
            f"но найдено только {len(best_peaks)}. "
            "Используем найденные пики."
        )
        count = len(best_peaks)

    # Выбираем самые выраженные пики.
    strongest = np.argsort(
        mean_envelope[best_peaks]
    )[-count:]

    seeds = np.sort(best_peaks[strongest])

    print("Выбранные глубины:", seeds)

    return seeds

def smooth_surface(surface, sigma=1.0):
    """Сглаживает горизонт, не распространяя NaN."""

    valid = np.isfinite(surface)

    numerator = gaussian_filter(
        np.where(valid, surface, 0.0),
        sigma=sigma
    )

    denominator = gaussian_filter(
        valid.astype(np.float32),
        sigma=sigma
    )

    result = np.full_like(surface, np.nan, dtype=np.float32)

    mask = denominator > 1e-6

    result[mask] = numerator[mask] / denominator[mask]

    return result


def make_seed_windows(seeds, depth_size, search_radius):
    """
    Создаёт непересекающиеся интервалы поиска.
    Каждый горизонт ищется только в своей области глубин.
    """

    seeds = np.asarray(seeds, dtype=int)

    windows = []

    for h, seed in enumerate(seeds):

        if h == 0:
            left = 0
        else:
            left = (seeds[h - 1] + seed) // 2

        if h == len(seeds) - 1:
            right = depth_size
        else:
            right = (seed + seeds[h + 1]) // 2 + 1

        lo = max(left, seed - search_radius)
        hi = min(right, seed + search_radius + 1)

        if hi <= lo:
            raise ValueError(
                f"Пустой интервал поиска для горизонта {h}: "
                f"{lo}:{hi}"
            )

        windows.append((lo, hi))

    return windows


def pick_horizons(
    seismic,
    seeds,
    search_radius=12,
    spatial_sigma=1.0,
    refine_radius=4
):
    """
    Поиск горизонтов с учётом пространственной согласованности.

    Результат имеет форму (H, C, I).
    """

    crossline_size, depth_size, inline_size = seismic.shape

    windows = make_seed_windows(
        seeds,
        depth_size,
        search_radius
    )

    print("Интервалы поиска:")

    for h, (lo, hi) in enumerate(windows):
        print(f"H{h + 1}: depth {lo}:{hi}")

    horizons = np.full(
        (len(seeds), crossline_size, inline_size),
        np.nan,
        dtype=np.float32
    )

    # Первый проход: независимый поиск пиков.
    for c in range(crossline_size):

        if c % 50 == 0:
            print(f"Первичный поиск: crossline {c}/{crossline_size}")

        traces = np.asarray(
            seismic[c],
            dtype=np.float32
        )

        valid_traces = np.isfinite(traces).all(axis=0)

        if not valid_traces.any():
            continue

        safe_traces = np.where(
            np.isfinite(traces),
            traces,
            0.0
        )

        envelope = np.abs(
            hilbert(safe_traces, axis=0)
        )

        envelope = gaussian_filter1d(
            envelope,
            sigma=1.0,
            axis=0
        )

        for h, (lo, hi) in enumerate(windows):

            local = envelope[lo:hi]

            picks = lo + np.argmax(
                local,
                axis=0
            )

            horizons[h, c, valid_traces] = picks[valid_traces]

    # Второй проход: медианное сглаживание по соседним трассам.
    smoothed = np.empty_like(horizons)

    for h in range(len(seeds)):

        valid = np.isfinite(horizons[h])

        safe = np.where(
            valid,
            horizons[h],
            float(seeds[h])
        )

        smoothed[h] = median_filter(
            safe,
            size=(3, 5),
            mode="nearest"
        )

        smoothed[h][~valid] = np.nan

    # Третий проход: уточняем положение на локальном отражении.
    refined = np.full_like(horizons, np.nan)

    offsets = np.arange(
        -refine_radius,
        refine_radius + 1
    )

    inline_indices = np.arange(inline_size)

    for c in range(crossline_size):

        if c % 50 == 0:
            print(f"Уточнение: crossline {c}/{crossline_size}")

        traces = np.asarray(
            seismic[c],
            dtype=np.float32
        )

        valid_traces = np.isfinite(traces).all(axis=0)

        if not valid_traces.any():
            continue

        safe_traces = np.where(
            np.isfinite(traces),
            traces,
            0.0
        )

        envelope = np.abs(
            hilbert(safe_traces, axis=0)
        )

        envelope = gaussian_filter1d(
            envelope,
            sigma=1.0,
            axis=0
        )

        for h, (lo, hi) in enumerate(windows):

            centers = np.rint(
                smoothed[h, c]
            ).astype(int)

            candidates = (
                centers[None, :]
                + offsets[:, None]
            )

            candidates = np.clip(
                candidates,
                lo,
                hi - 1
            )

            scores = np.take_along_axis(
                envelope,
                candidates,
                axis=0
            )

            best = np.argmax(
                scores,
                axis=0
            )

            picks = candidates[
                best,
                inline_indices
            ]

            refined[h, c, valid_traces] = (
                picks[valid_traces]
            )

    # Последнее небольшое сглаживание.
    if spatial_sigma > 0:

        for h in range(len(seeds)):

            refined[h] = smooth_surface(
                refined[h],
                sigma=spatial_sigma
            )

    return refined

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark-dir",
        default="../pytorch-3dunet-master/my_data/benchmark"
    )

    parser.add_argument(
        "--output-dir",
        default="rgt_work"
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help="Имя HDF5 dataset, обычно raw. Если не задано, скрипт ищет автоматически."
    )

    parser.add_argument(
        "--channel",
        type=int,
        default=0
    )

    parser.add_argument(
        "--count",
        type=int,
        default=8
    )

    parser.add_argument(
        "--search-radius",
        type=int,
        default=12
    )

    parser.add_argument(
        "--spatial-sigma",
        type=float,
        default=1.0
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seismic_path = output_dir / "full_seismic.npy"

    if not seismic_path.exists():
        assemble_full_seismic(
            args.benchmark_dir,
            seismic_path,
            dataset_name=args.dataset,
            channel=args.channel
        )
    else:
        print("Используем уже собранную сейсмику:", seismic_path)

    seismic = np.load(
        seismic_path,
        mmap_mode="r"
    )

    if seismic.shape != FULL_SHAPE:
        raise ValueError(
            f"Неверная форма полной сейсмики: {seismic.shape}"
        )

    print("\nПолная сейсмика:", seismic.shape)

    # 2. Выбираем seed depths только по TRAIN.
    train_seismic = seismic[0:631, :, 200:601]

    seeds = choose_seed_depths(
        train_seismic,
        count=args.count
    )

    del train_seismic

    print("\nВыбранные seed depths:", seeds)

    # 3. Ищем горизонты по всему объёму.
    horizons = pick_horizons(
        seismic,
        seeds,
        search_radius=args.search_radius,
        spatial_sigma=args.spatial_sigma
    )

    output_path = output_dir / "horizons_seismic.npz"

    np.savez_compressed(
        output_path,
        depth=horizons,
        seed_depths=seeds,
        axis_order="H,C,I"
    )

    print("\n========== HORIZON QC ==========")
    print("Horizons shape:", horizons.shape)
    print(
        "Coverage:",
        round(np.isfinite(horizons).mean() * 100, 3),
        "%"
    )
    print("Seed depths:", seeds)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()