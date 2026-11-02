
from pathlib import Path
import json

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt


# =========================================================
# 1. PATHS
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

MY_DATA = (
    PROJECT_DIR
    / "pytorch-3dunet-master"
    / "my_data"
)

RAW_DIR = MY_DATA / "alaudah_raw"
HORIZONS_DIR = RAW_DIR / "Horizons"
BENCHMARK_DIR = MY_DATA / "benchmark"

LABELS_FILE = RAW_DIR / "labels_entire_volume.npy"

OUTPUT_DIR = (
    SCRIPT_DIR
    / "rgt_output"
    / "alaudah"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. GEOMETRY
# =========================================================

N_CROSSLINES = 901
N_INLINES = 601
N_DEPTH = 255

FIRST_CROSSLlNE = 300
FIRST_INLINE = 100

DEPTH_MIN = 1005.0
DEPTH_MAX = 1877.0

DEPTH_STEP = (
    DEPTH_MAX - DEPTH_MIN
) / (N_DEPTH - 1)

# Пространственная сетка:
# [crossline, inline]
#
# Полный диапазон:
# crossline 300..1200
# inline 100..700


# =========================================================
# 3. HORIZONS IN GEOLOGICAL ORDER
# =========================================================

# Порядок сверху вниз.
# Chalk и Rijnland остаются отдельными поверхностями,
# хотя в разметке они объединены в один класс.

HORIZON_ORDER = [
    "Upper North Sea Group",
    "Middle North Sea Group",
    "Lowe North Sea Group",
    "Chalk Group",
    "Rijnland Group",
    "Scruff Group",
    "Zechstein Group",
]

# Максимальное расстояние для заполнения небольших
# пропусков ближайшими интерпретированными точками.
MAX_HORIZON_FILL_DISTANCE = 2.0

# Максимальное допустимое исправление пересечения
# поверхностей вдоль одной трассы, в отсчётах глубины.
MAX_CROSSING_CORRECTION = 8.0

# Допустимое расстояние для заполнения трасс,
# где не хватает контрольных поверхностей.
MAX_PROFILE_FILL_DISTANCE = 40


# =========================================================
# 4. HELPERS
# =========================================================

def section(title):
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def pearson_distance_to_boundary(trace, depth_index):
    """
    Расстояние от заданной глубины до ближайшей
    границы классов в трассе разметки.
    """

    transitions = (
        np.flatnonzero(np.diff(trace) != 0) + 1
    )

    if transitions.size == 0:
        return np.nan

    return float(
        np.min(np.abs(transitions - depth_index))
    )


# =========================================================
# 5. READ AND MAP HORIZONS
# =========================================================

def read_horizons():

    section("ЧТЕНИЕ ГЕОЛОГИЧЕСКИХ ГОРИЗОНТОВ")

    surfaces = {}
    surface_stats = {}

    for rank, name in enumerate(HORIZON_ORDER):

        path = HORIZONS_DIR / name

        if not path.exists():
            print("Не найден:", path)
            continue

        print(f"\n{name}")

        data = np.loadtxt(path, dtype=np.float64)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] != 5:
            raise ValueError(
                f"{name}: ожидалось 5 столбцов, "
                f"получено {data.shape[1]}"
            )

        inline = data[:, 0]
        crossline = data[:, 1]

        x = data[:, 2]
        y = data[:, 3]
        z = data[:, 4]

        # Проверяем, что Inline/Crossline — целые номера.
        coordinate_error = max(
            float(np.max(np.abs(inline - np.rint(inline)))),
            float(np.max(np.abs(crossline - np.rint(crossline)))),
        )

        if coordinate_error > 1e-4:
            raise ValueError(
                f"{name}: Inline/Crossline не являются "
                "целыми координатами сетки."
            )

        inline = np.rint(inline).astype(np.int32)
        crossline = np.rint(crossline).astype(np.int32)

        # Переводим реальные номера в индексы.
        i = inline - FIRST_INLINE
        c = crossline - FIRST_CROSSLlNE

        spatial_valid = (
            (i >= 0)
            & (i < N_INLINES)
            & (c >= 0)
            & (c < N_CROSSLINES)
            & np.isfinite(z)
        )

        # Используем только точки внутри вертикального
        # диапазона benchmark.
        depth_valid = (
            (z >= DEPTH_MIN)
            & (z <= DEPTH_MAX)
        )

        valid = spatial_valid & depth_valid

        print("Всего точек:", len(data))
        print("Внутри пространственной области:",
              int(spatial_valid.sum()))
        print("Внутри глубинного диапазона:",
              int(valid.sum()))

        if valid.sum() < 1000:
            print("Поверхность пропускается: недостаточно точек.")
            continue

        # -------------------------------------------------
        # Проверка X/Y
        # -------------------------------------------------

        # Начало сетки и направления осей,
        # ранее восстановленные по координатам F3.
        origin = np.array([
            605835.47552005,
            6073556.33478774,
        ])

        v_crossline = np.array([
            24.99031579,
            0.69852632,
        ])

        v_inline = np.array([
            -0.69811617,
            24.98963893,
        ])

        predicted_xy = (
            origin
            + c[spatial_valid, None] * v_crossline
            + i[spatial_valid, None] * v_inline
        )

        actual_xy = np.column_stack([
            x[spatial_valid],
            y[spatial_valid],
        ])

        xy_errors = np.linalg.norm(
            actual_xy - predicted_xy,
            axis=1,
        )

        xy_p95 = float(np.percentile(xy_errors, 95))

        print("Ошибка XY, 95-й перцентиль:",
              f"{xy_p95:.3f} м")

        if xy_p95 > 2.0:
            raise RuntimeError(
                f"{name}: XY не совпадают с ожидаемой сеткой."
            )

        # -------------------------------------------------
        # Переводим Z в вертикальные индексы.
        # -------------------------------------------------

        depth_index = (
            (z[valid] - DEPTH_MIN) / DEPTH_STEP
        )

        c_valid = c[valid]
        i_valid = i[valid]

        # Растеризуем разреженные точки в сетку.
        # Если есть дубликаты координат, усредняем их.
        flat_indices = (
            c_valid.astype(np.int64) * N_INLINES
            + i_valid.astype(np.int64)
        )

        
        sums = np.bincount(
            flat_indices,
            weights=depth_index,
            minlength=N_CROSSLINES * N_INLINES,
        ).reshape(N_CROSSLINES, N_INLINES)

        counts = np.bincount(
            flat_indices,
            minlength=N_CROSSLINES * N_INLINES,
        ).reshape(N_CROSSLINES, N_INLINES)

        grid = np.full(
            (N_CROSSLINES, N_INLINES),
            np.nan,
            dtype=np.float32,
        )

        mask = counts > 0

        grid[mask] = (
            sums[mask] / counts[mask]
        ).astype(np.float32)


        unique_positions = int(mask.sum())

        print("Уникальных точек сетки:", unique_positions)
        print(
            "Покрытие полного куба:",
            f"{100 * unique_positions / mask.size:.2f}%",
        )

        # -------------------------------------------------
        # Заполняем небольшие пропуски по горизонтали.
        # -------------------------------------------------

        distance, nearest_indices = distance_transform_edt(
            ~mask,
            return_indices=True,
        )

        filled = grid[tuple(nearest_indices)]

        filled[distance > MAX_HORIZON_FILL_DISTANCE] = np.nan

        final_mask = np.isfinite(filled)

        print(
            "Покрытие после локального заполнения:",
            f"{100 * final_mask.mean():.2f}%",
        )

        surfaces[name] = {
            "depth": filled,
            "rank": rank,
        }

        surface_stats[name] = {
            "points_total": int(len(data)),
            "points_inside_volume": int(valid.sum()),
            "unique_grid_points": unique_positions,
            "coverage_before_fill": float(mask.mean()),
            "coverage_after_fill": float(final_mask.mean()),
            "xy_error_p95_m": xy_p95,
            "z_min_inside": float(z[valid].min()),
            "z_max_inside": float(z[valid].max()),
        }

    if len(surfaces) < 3:
        raise RuntimeError(
            "Получено меньше трёх пригодных поверхностей. "
            "Нельзя строить надёжный RGT."
        )

    return surfaces, surface_stats


# =========================================================
# 6. VALIDATE Z AGAINST TRAIN/VAL LABEL BOUNDARIES
# =========================================================

def validate_depth_mapping(surface_stats):

    section("ПРОВЕРКА ВЕРТИКАЛЬНОГО СОВМЕЩЕНИЯ ПО LABEL")

    train_path = BENCHMARK_DIR / "train.h5"
    val_path = BENCHMARK_DIR / "val.h5"

    results = {}

    rng = np.random.default_rng(42)

    with h5py.File(train_path, "r") as train_file, \
         h5py.File(val_path, "r") as val_file:

        train_labels = train_file["label"]
        val_labels = val_file["label"]

        # Валидация использует только Train и Val.
        # Test1/Test2 здесь не читаются.

        for name in HORIZON_ORDER:

            path = HORIZONS_DIR / name

            if not path.exists():
                continue

            data = np.loadtxt(path, dtype=np.float64)

            if data.ndim == 1:
                data = data.reshape(1, -1)

            inline = np.rint(data[:, 0]).astype(int)
            crossline = np.rint(data[:, 1]).astype(int)
            z = data[:, 4]

            i = inline - FIRST_INLINE
            c = crossline - FIRST_CROSSLlNE

            valid = (
                (i >= 200)
                & (i < 601)
                & (c >= 0)
                & (c < 701)
                & (z >= DEPTH_MIN)
                & (z <= DEPTH_MAX)
            )

            indices = np.flatnonzero(valid)

            if len(indices) > 3000:
                indices = rng.choice(
                    indices,
                    size=3000,
                    replace=False,
                )

            distances = []

            for idx in indices:

                ci = c[idx]
                ii = i[idx] - 200

                j = (
                    z[idx] - DEPTH_MIN
                ) / DEPTH_STEP

                if ci < 631:
                    trace = np.asarray(
                        train_labels[ci, :, ii]
                    )
                else:
                    trace = np.asarray(
                        val_labels[ci - 631, :, ii]
                    )

                distance = pearson_distance_to_boundary(
                    trace,
                    j,
                )

                if np.isfinite(distance):
                    distances.append(distance)

            if not distances:
                print(f"{name}: нет подходящих точек Train/Val")
                continue

            distances = np.asarray(distances)

            median = float(np.median(distances))
            p90 = float(np.percentile(distances, 90))
            near_fraction = float(
                np.mean(distances <= 3)
            )

            results[name] = {
                "samples": int(len(distances)),
                "median_distance_samples": median,
                "p90_distance_samples": p90,
                "fraction_within_3_samples": near_fraction,
            }

            print(
                f"{name}: "
                f"median={median:.2f}, "
                f"p90={p90:.2f}, "
                f"within 3 samples={near_fraction:.1%}"
            )

    # Это контроль согласованности, а не требование,
    # чтобы каждая поверхность обязательно совпадала
    # с границей класса. Chalk/Rijnland разделяют один класс.
    good = sum(
        item["fraction_within_3_samples"] >= 0.30
        for item in results.values()
        if item["samples"] >= 100
    )

    print("\nПоверхностей с заметным совпадением границ:", good)

    if good < 2:
        raise RuntimeError(
            "Проверка Z по границам разметки не подтвердила "
            "вертикальное соответствие. RGT не создаётся."
        )

    return results


# =========================================================
# 7. BUILD RGT
# =========================================================

def build_rgt(surfaces):

    section("ПОСТРОЕНИЕ RGT")

    # Полный RGT во временном порядке:
    # [crossline, inline, depth]
    rgt = np.full(
        (N_CROSSLINES, N_INLINES, N_DEPTH),
        np.nan,
        dtype=np.float32,
    )

    valid_columns = np.zeros(
        (N_CROSSLINES, N_INLINES),
        dtype=bool,
    )

    max_shift_seen = 0.0
    corrected_columns = 0
    rejected_columns = 0

    depth_axis = np.arange(
        N_DEPTH,
        dtype=np.float32,
    )

    names = [
        name for name in HORIZON_ORDER
        if name in surfaces
    ]

    print("Используемые поверхности:")
    for name in names:
        print(" ", name)

    for c in range(N_CROSSLINES):

        if (c + 1) % 50 == 0:
            print(
                f"Обработка crossline {c + 1}/{N_CROSSLINES}",
                end="\r",
                flush=True,
            )

        for i in range(N_INLINES):

            positions = [0.0]
            values = [0.0]

            # Добавляем поверхности сверху вниз.
            for name in names:

                surface = surfaces[name]["depth"]

                j = surface[c, i]

                if not np.isfinite(j):
                    continue

                positions.append(float(j))

                # Фиксированные стратиграфические уровни:
                # Upper=0, Middle=1, Lower=2, Chalk=3,
                # Rijnland=4, Scruff=5, Zechstein=6.
                values.append(
                    surfaces[name]["rank"] / 6.0
                )

            if len(positions) < 2:
                rejected_columns += 1
                continue

            positions = np.asarray(
                positions,
                dtype=np.float64,
            )

            values = np.asarray(
                values,
                dtype=np.float64,
            )

            # Исправляем небольшие пересечения горизонтов,
            # сохраняя заданный стратиграфический порядок.
            fixed = np.maximum.accumulate(
                positions + np.arange(len(positions)) * 0.01
            )

            correction = fixed - positions

            max_shift = float(correction.max())

            max_shift_seen = max(
                max_shift_seen,
                max_shift,
            )

            if max_shift > MAX_CROSSING_CORRECTION:
                rejected_columns += 1
                continue

            if max_shift > 0.5:
                corrected_columns += 1

            profile = np.interp(
                depth_axis,
                fixed,
                values,
                left=values[0],
                right=values[-1],
            )

            rgt[c, i, :] = profile.astype(np.float32)

            valid_columns[c, i] = True

    print()

    valid_fraction = float(valid_columns.mean())

    print("Трасс построено:", int(valid_columns.sum()))
    print("Трасс с пересечениями, исправленными локально:",
          corrected_columns)
    print("Трасс отвергнуто:", rejected_columns)
    print("Максимальное исправление:",
          f"{max_shift_seen:.3f} отсчёта")
    print("Доля построенных трасс:",
          f"{100 * valid_fraction:.2f}%")

    if valid_fraction < 0.85:
        raise RuntimeError(
            "Слишком мало трасс получили RGT. "
            "Не заполняем большой объём догадками."
        )

    # Заполняем оставшиеся трассы ближайшими
    # построенными профилями по горизонтали.
    distance, nearest = distance_transform_edt(
        ~valid_columns,
        return_indices=True,
    )

    if np.any(~valid_columns):

        max_distance = float(
            distance[~valid_columns].max()
        )

        print(
            "Максимальное расстояние до построенной трассы:",
            f"{max_distance:.2f} бинов",
        )

        if max_distance > MAX_PROFILE_FILL_DISTANCE:
            raise RuntimeError(
                "Есть большие области без достаточного "
                "числа горизонтов. Автоматическое заполнение "
                "небезопасно."
            )

        missing = ~valid_columns

        rgt[missing] = rgt[
            nearest[0][missing],
            nearest[1][missing],
            :
        ]

    # Проверяем монотонность RGT по глубине.
    differences = np.diff(rgt, axis=2)

    monotonic_fraction = float(
        np.mean(differences >= -1e-6)
    )

    print(
        "Монотонность RGT по глубине:",
        f"{100 * monotonic_fraction:.4f}%",
    )

    if monotonic_fraction < 0.999:
        raise RuntimeError(
            "RGT оказался немонотонным по глубине."
        )

    return rgt


# =========================================================
# 8. SAVE FOUR BENCHMARK VOLUMES
# =========================================================

def save_splits(rgt):

    section("СОХРАНЕНИЕ RGT ДЛЯ BENCHMARK")

    # Сейчас rgt: [crossline, inline, depth]
    #
    # Для модели нужно:
    # [crossline, depth, inline]

    splits = {
        "train": (
            rgt[0:631, 200:601, :]
        ),
        "val": (
            rgt[631:701, 200:601, :]
        ),
        "test1": (
            rgt[0:701, 0:200, :]
        ),
        "test2": (
            rgt[701:901, 0:601, :]
        ),
    }

    expected = {
        "train": (631, 255, 401),
        "val": (70, 255, 401),
        "test1": (701, 255, 200),
        "test2": (200, 255, 601),
    }

    for name, volume in splits.items():

        volume = np.transpose(
            volume,
            (0, 2, 1),
        ).copy()

        if volume.shape != expected[name]:
            raise RuntimeError(
                f"{name}: неправильная форма {volume.shape}"
            )

        output_path = OUTPUT_DIR / f"rgt_{name}.npy"

        np.save(
            output_path,
            volume.astype(np.float32),
        )

        print(f"\n{name}:")
        print("  shape:", volume.shape)
        print("  min:", float(volume.min()))
        print("  max:", float(volume.max()))
        print("  mean:", float(volume.mean()))
        print("  saved:", output_path)


# =========================================================
# 9. MAIN
# =========================================================

def main():

    section("ALAUDАH RGT CONSTRUCTION")

    required = [
        RAW_DIR,
        HORIZONS_DIR,
        LABELS_FILE,
        BENCHMARK_DIR / "train.h5",
        BENCHMARK_DIR / "val.h5",
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    print("Depth range:", DEPTH_MIN, "to", DEPTH_MAX, "m")
    print("Depth step:", f"{DEPTH_STEP:.6f} m")
    print("Full geometry:", (901, 255, 601))

    # 1. Читаем и отображаем горизонты.
    surfaces, surface_stats = read_horizons()

    # 2. Проверяем соответствие Z разметке.
    label_validation = validate_depth_mapping(
        surface_stats
    )

    # 3. Строим RGT.
    rgt = build_rgt(surfaces)

    # 4. Сохраняем четыре части.
    save_splits(rgt)

    # 5. Сохраняем метаданные.
    metadata = {
        "input_seismic": str(
            RAW_DIR / "seismic_entire_volume.npy"
        ),
        "input_labels": str(LABELS_FILE),
        "horizons_dir": str(HORIZONS_DIR),
        "axis_order_output": "crossline, depth, inline",
        "depth_min_m": DEPTH_MIN,
        "depth_max_m": DEPTH_MAX,
        "depth_samples": N_DEPTH,
        "depth_step_m": DEPTH_STEP,
        "rgt_method": (
            "Piecewise-linear interpolation between "
            "interpreted stratigraphic horizons"
        ),
        "horizon_order": HORIZON_ORDER,
        "surface_statistics": surface_stats,
        "label_boundary_validation": label_validation,
        "max_horizon_fill_distance": MAX_HORIZON_FILL_DISTANCE,
        "max_crossing_correction": MAX_CROSSING_CORRECTION,
    }

    metadata_path = (
        OUTPUT_DIR / "rgt_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False,
        )

    section("ГОТОВО")

    print("RGT сохранён в:", OUTPUT_DIR)
    print("Метаданные:", metadata_path)


if __name__ == "__main__":
    main()
