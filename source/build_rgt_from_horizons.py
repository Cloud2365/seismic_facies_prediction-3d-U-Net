import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


def fill_missing_horizon(surface, max_distance=40):
    """
    Заполняет небольшие пропуски ближайшими точками горизонта.

    surface: (crossline, inline)
    """

    valid = np.isfinite(surface)

    if not valid.any():
        raise ValueError(
            "Горизонт полностью пустой."
        )

    if valid.all():
        return surface.copy(), np.zeros(
            surface.shape,
            dtype=np.float32
        )

    distances, indices = distance_transform_edt(
        ~valid,
        return_indices=True
    )

    filled = surface[tuple(indices)].astype(np.float32)

    filled[distances > max_distance] = np.nan

    return filled, distances.astype(np.float32)


def order_horizons(horizons):
    """Сортирует горизонты по медианной глубине."""

    medians = np.nanmedian(
        horizons,
        axis=(1, 2)
    )

    if not np.isfinite(medians).all():
        raise ValueError(
            "У одного или нескольких горизонтов нет корректной медианной глубины."
        )

    order = np.argsort(medians)

    return horizons[order], medians[order], order


def enforce_trace_order(positions, min_gap=1.0):
    """
    Исправляет локальные пересечения на одной трассе.

    positions: (H,)
    """

    original = positions.copy()
    adjusted = original.copy()

    for j in range(1, len(adjusted)):
        adjusted[j] = max(
            adjusted[j],
            adjusted[j - 1] + min_gap
        )

    correction = float(
        np.max(np.abs(adjusted - original))
    )

    return adjusted, correction


def build_rgt(
    horizons,
    depth_size=255,
    max_fill_distance=40,
    min_gap=1.0,
    max_correction=8.0,
    output_path=None
):
    """
    Строит RGT с формой (crossline, depth, inline).

    Все принятые трассы получают RGT от 0 до 1.
    """

    if horizons.ndim != 3:
        raise ValueError(
            f"Ожидается (H,C,I), получено {horizons.shape}"
        )

    horizon_count, crossline_size, inline_size = horizons.shape

    # Сохраняем исходное покрытие до заполнения.
    input_coverage = float(
        np.isfinite(horizons).mean()
    )

    # Сортировка горизонтов.
    horizons, medians, order = order_horizons(horizons)

    print("Порядок горизонтов:", order)
    print("Медианные глубины:", medians)

    # Заполнение пропусков.
    filled = np.full_like(
        horizons,
        np.nan,
        dtype=np.float32
    )

    fill_distances = np.zeros_like(
        horizons,
        dtype=np.float32
    )

    missing_mask = ~np.isfinite(horizons)

    for h in range(horizon_count):

        filled[h], fill_distances[h] = fill_missing_horizon(
            horizons[h],
            max_distance=max_fill_distance
        )

    filled_coverage = float(
        np.isfinite(filled).mean()
    )

    if output_path is None:
        raise ValueError("Не задан путь для сохранения RGT.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Создаём RGT на диске, а не целиком в оперативной памяти.
    rgt = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(crossline_size, depth_size, inline_size)
    )

    rgt[:] = np.nan

    # Значения RGT на горизонтах.
    rgt_levels = np.arange(
        1,
        horizon_count + 1,
        dtype=np.float64
    ) / (horizon_count + 1)

    rejected_missing = 0
    rejected_order = 0
    rejected_bounds = 0
    corrected_traces = 0
    accepted_traces = 0

    max_seen_correction = 0.0

    for c in range(crossline_size):

        if c % 50 == 0:
            print(f"Построение RGT: crossline {c}/{crossline_size}")

        # Форма (H, I).
        positions = filled[:, c, :].copy()

        valid = np.isfinite(positions).all(axis=0)

        rejected_missing += int((~valid).sum())

        if not valid.any():
            continue

        # Работаем сразу с группой трасс.
        selected = positions[:, valid]

        adjusted = selected.copy()

        for h in range(1, horizon_count):
            adjusted[h] = np.maximum(
                adjusted[h],
                adjusted[h - 1] + min_gap
            )

        corrections = np.max(
            np.abs(adjusted - selected),
            axis=0
        )

        if corrections.size:
            max_seen_correction = max(
                max_seen_correction,
                float(np.max(corrections))
            )

        corrected_traces += int(
            np.sum(corrections > 1e-6)
        )

        # Слишком большие исправления не принимаем.
        good_order = corrections <= max_correction

        rejected_order += int(
            (~good_order).sum()
        )

        # Горизонты должны находиться строго внутри объёма,
        # чтобы не совпадать с верхней и нижней границами.
        good_bounds = (
            (adjusted[0] > 0)
            & (adjusted[-1] < depth_size - 1)
        )

        rejected_bounds += int(
            (good_order & ~good_bounds).sum()
        )

        good = good_order & good_bounds

        good_indices = np.flatnonzero(valid)

        for local_index in np.flatnonzero(good):

            inline_index = good_indices[local_index]

            trace_positions = adjusted[:, local_index]

            x = np.concatenate([
                [0.0],
                trace_positions.astype(np.float64),
                [float(depth_size - 1)]
            ])

            # Обязательно проверяем строгий порядок.
            if not np.all(np.diff(x) > 0):
                rejected_bounds += 1
                continue

            y = np.concatenate([
                [0.0],
                rgt_levels,
                [1.0]
            ])

            rgt[c, :, inline_index] = np.interp(
                np.arange(depth_size),
                x,
                y
            ).astype(np.float32)

            accepted_traces += 1

    rgt.flush()

    total_traces = crossline_size * inline_size

    valid_trace_fraction = accepted_traces / total_traces

    # Статистика расстояний заполнения только в пропущенных точках.
    missing_distances = fill_distances[missing_mask]

    if missing_distances.size:
        mean_fill_distance = float(
            np.mean(missing_distances)
        )
        max_fill_distance_seen = float(
            np.max(missing_distances)
        )
    else:
        mean_fill_distance = 0.0
        max_fill_distance_seen = 0.0

    qc = {
        "input_coverage": input_coverage,
        "filled_coverage": filled_coverage,
        "mean_fill_distance": mean_fill_distance,
        "max_fill_distance_seen": max_fill_distance_seen,
        "valid_trace_fraction": valid_trace_fraction,
        "accepted_traces": accepted_traces,
        "rejected_missing": rejected_missing,
        "rejected_order": rejected_order,
        "rejected_bounds": rejected_bounds,
        "corrected_traces": corrected_traces,
        "max_order_correction": max_seen_correction,
        "rgt_min": 0.0 if accepted_traces else float("nan"),
        "rgt_max": 1.0 if accepted_traces else float("nan"),
    }

    return rgt, filled, qc, order


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--horizons",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        default="rgt_work"
    )

    parser.add_argument(
        "--depth-size",
        type=int,
        default=255
    )

    parser.add_argument(
        "--max-fill-distance",
        type=float,
        default=40
    )

    parser.add_argument(
        "--min-gap",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--max-correction",
        type=float,
        default=8.0
    )

    args = parser.parse_args()

    data = np.load(args.horizons)

    if "depth" not in data:
        raise ValueError(
            "В NPZ отсутствует массив depth."
        )

    original_horizons = np.asarray(
        data["depth"],
        dtype=np.float32
    )

    seeds = (
        data["seed_depths"]
        if "seed_depths" in data
        else np.array([], dtype=np.int16)
    )

    print("Исходная форма горизонтов:", original_horizons.shape)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgt_path = output_dir / "rgt.npy"

    rgt, filled, qc, order = build_rgt(
        original_horizons,
        depth_size=args.depth_size,
        max_fill_distance=args.max_fill_distance,
        min_gap=args.min_gap,
        max_correction=args.max_correction,
        output_path=rgt_path
    )

    qc_path = output_dir / "rgt_qc.npz"

    np.savez_compressed(
        qc_path,
        horizons_original=original_horizons,
        horizons_filled=filled,
        seed_depths=seeds,
        horizon_order=order,
        **qc
    )

    print("\n========== RGT QC ==========")

    for key, value in qc.items():
        print(f"{key}: {value}")

    print("\nRGT shape:", rgt.shape)
    print("RGT saved:", rgt_path)
    print("QC saved:", qc_path)


if __name__ == "__main__":
    main()