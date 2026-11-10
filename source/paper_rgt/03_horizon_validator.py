
import os
import sys
import json
import subprocess
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

SOURCE = os.path.join(
    BASE, "rgt_work", "mhe_work", "extracted", "horizons_mhe.npz"
)
OUTPUT = os.path.join(
    BASE, "rgt_work", "mhe_work", "extracted", "horizons_ordered.npz"
)
RGT_DIR = os.path.join(
    BASE, "rgt_work", "mhe_work", "rgt_ordered"
)
META = os.path.join(
    BASE, "rgt_work", "mhe_work", "volume_meta.json"
)
QC_OUTPUT = os.path.join(
    RGT_DIR, "correction_report.json"
)

MIN_GAP = 1.0
DEPTH_MIN = 0.0
DEPTH_MAX = 254.0


def project_trace(values):
    """
    Находит минимальное суммарное изменение шести горизонтов,
    обеспечивая их порядок, минимальный зазор и границы глубины.
    Используется bounded isotonic regression (PAVA).
    """
    n = len(values)
    indices = np.arange(n, dtype=float)

    # Преобразование: после него нужен неубывающий ряд.
    z = values - indices * MIN_GAP

    blocks = []

    for i in range(n):
        lower = DEPTH_MIN - i * MIN_GAP
        upper = DEPTH_MAX - i * MIN_GAP

        block = {
            "start": i,
            "end": i,
            "sum": z[i],
            "count": 1,
            "lower": lower,
            "upper": upper,
        }
        block["value"] = np.clip(z[i], lower, upper)
        blocks.append(block)

        # Объединяем соседние блоки, если порядок нарушен.
        while (
            len(blocks) >= 2
            and blocks[-2]["value"] > blocks[-1]["value"]
        ):
            b = blocks.pop()
            a = blocks.pop()

            merged = {
                "start": a["start"],
                "end": b["end"],
                "sum": a["sum"] + b["sum"],
                "count": a["count"] + b["count"],
                "lower": max(a["lower"], b["lower"]),
                "upper": min(a["upper"], b["upper"]),
            }

            mean = merged["sum"] / merged["count"]
            merged["value"] = np.clip(
                mean, merged["lower"], merged["upper"]
            )
            blocks.append(merged)

    result = np.empty(n, dtype=np.float64)

    for block in blocks:
        result[block["start"]:block["end"] + 1] = block["value"]

    return result + indices * MIN_GAP


def main():
    if not os.path.exists(SOURCE):
        raise FileNotFoundError(SOURCE)

    print("Загрузка исходных горизонтов...")
    archive = np.load(SOURCE, allow_pickle=True)
    data = {key: archive[key] for key in archive.files}
    archive.close()

    horizons = data["horizons"].copy()
    confidence = data["confidence"].copy()

    if horizons.shape[0] != 6:
        raise ValueError(
            f"Ожидалось 6 горизонтов, получено: {horizons.shape}"
        )

    original = horizons.copy()
    count, crosslines, inlines = horizons.shape

    print(f"Размер поверхностей: {horizons.shape}")
    print("Проверка исходных пересечений...")

    before = {}
    for i in range(5):
        valid = np.isfinite(original[i]) & np.isfinite(original[i + 1])
        gap = original[i + 1] - original[i]
        before[f"H{i+1:02d}_H{i+2:02d}"] = {
            "crossings": int(np.count_nonzero(valid & (gap <= 0))),
            "gap_below_one": int(np.count_nonzero(valid & (gap < 1))),
        }

    flat = horizons.reshape(6, -1)
    original_flat = original.reshape(6, -1)

    valid_traces = np.all(np.isfinite(flat), axis=0)
    valid_indices = np.flatnonzero(valid_traces)

    print(f"Трасс для коррекции: {len(valid_indices)}")

    correction_mask = np.zeros(
        (6, flat.shape[1]), dtype=bool
    )

    # Корректируем только трассы, где есть нарушение порядка
    # или выход за пределы глубины.
    for number, trace_idx in enumerate(valid_indices):
        values = flat[:, trace_idx]

        gaps = np.diff(values)
        needs_fix = (
            np.any(gaps < MIN_GAP)
            or np.any(values < DEPTH_MIN)
            or np.any(values > DEPTH_MAX)
        )

        if needs_fix:
            corrected = project_trace(values)
            flat[:, trace_idx] = corrected
            correction_mask[:, trace_idx] = (
                np.abs(corrected - values) > 1e-6
            )

        if (number + 1) % 100000 == 0:
            print(f"Обработано: {number + 1}/{len(valid_indices)}")

    horizons = flat.reshape(6, crosslines, inlines)
    correction_mask = correction_mask.reshape(
        6, crosslines, inlines
    )

    data["horizons"] = horizons
    data["correction_mask"] = correction_mask

    # Не меняем исходную confidence: сохраняем её смысл.
    np.savez_compressed(OUTPUT, **data)

    print("\n=== Результаты коррекции ===")

    report = {
        "method": "bounded isotonic regression",
        "min_gap": MIN_GAP,
        "depth_bounds": [DEPTH_MIN, DEPTH_MAX],
        "valid_traces": int(len(valid_indices)),
        "before": before,
        "horizons": {},
        "remaining_violations": {},
    }

    for i in range(6):
        diff = horizons[i] - original[i]
        changed = np.isfinite(diff) & (np.abs(diff) > 1e-6)
        values = np.abs(diff[np.isfinite(diff)])

        report["horizons"][f"H{i+1:02d}"] = {
            "changed_traces": int(np.count_nonzero(changed)),
            "mean_absolute_shift": float(np.mean(values)),
            "p95_absolute_shift": float(np.percentile(values, 95)),
            "max_absolute_shift": float(np.max(values)),
        }

        print(
            f"H{i+1:02d}: изменено {np.count_nonzero(changed)} трасс; "
            f"средний сдвиг {np.mean(values):.3f}; "
            f"максимальный {np.max(values):.3f}"
        )

    for i in range(5):
        valid = np.isfinite(horizons[i]) & np.isfinite(horizons[i + 1])
        gap = horizons[i + 1] - horizons[i]

        violations = int(np.count_nonzero(valid & (gap < MIN_GAP)))
        report["remaining_violations"][
            f"H{i+1:02d}_H{i+2:02d}"
        ] = violations

        print(
            f"H{i+1:02d} -> H{i+2:02d}: "
            f"нарушений порядка/зазора: {violations}"
        )

    os.makedirs(RGT_DIR, exist_ok=True)

    with open(QC_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nИсправленный архив: {OUTPUT}")

    print("\nПостроение нового RGT...")
    command = [
        sys.executable,
        os.path.join(BASE, "mhe_f3_toolkit", "build_rgt_from_mhe.py"),
        "--horizons", OUTPUT,
        "--meta", META,
        "--output-dir", RGT_DIR,
    ]

    subprocess.run(command, check=True)

    print("\n=== ГОТОВО ===")
    print(f"Новый RGT: {RGT_DIR}")
    print(f"Отчёт коррекции: {QC_OUTPUT}")


if __name__ == "__main__":
    main()
