#!/usr/bin/env python3
"""Проверка и коррекция горизонтов MHE перед построением RGT.

Входная папка (--work) должна содержать файлы, созданные ExtractHorizons.java:
  horizons_raw.bin  little-endian float32, порядок [horizon][crossline][inline]
  horizon_ids.txt   ID горизонтов в том же порядке, через запятую

volume_meta.json может лежать в --work или в родительской папке.
Его также можно указать явно через --meta.

Результат:
  horizons_ordered.bin          исправленные поверхности (исходный порядок ID)
  horizons_correction_mask.npy  маска глубин, изменённых коррекцией
  <rgt-dir>/horizon_correction_report.json
  <rgt-dir>/rgt.npy              если не указан --no-build

Исходный horizons_raw.bin не изменяется. Неполные трассы (хотя бы один
NaN/Inf среди горизонтов) не исправляются: они останутся невалидными для RGT.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parent
DEFAULT_WORK = SOURCE_DIR / "rgt_work" / "mhe_work"
BUILD_SCRIPT = SCRIPT_DIR / "04_build_rgt.py"
DEFAULT_MIN_GAP = 1.0
# Маленький запас нужен, потому что итоговые горизонты сохраняются как float32.
FLOAT32_GAP_MARGIN = 1.0e-4


def read_horizon_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл ID горизонтов: {path}")
    content = path.read_text(encoding="utf-8-sig").strip()
    ids = [item.strip() for item in content.replace("\n", "").split(",") if item.strip()]
    if not ids:
        raise ValueError(f"Файл {path} пуст: ID горизонтов не заданы.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"В {path} обнаружены повторяющиеся ID: {ids}")
    return ids


def resolve_meta_path(work: Path, explicit_meta: Path | None) -> Path:
    if explicit_meta is not None:
        path = explicit_meta.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Не найден volume_meta.json: {path}")
        return path

    # Частый вариант: Java пишет горизонты в папку extracted, а meta находится
    # на уровень выше, рядом с seismic_mhe.bin.
    candidates = [work / "volume_meta.json", work.parent / "volume_meta.json"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Не найден volume_meta.json. Положи его в --work или родительскую папку, "
        "либо передай явно: --meta путь\\к\\volume_meta.json"
    )


def finite_median(surface: np.ndarray) -> float:
    values = surface[np.isfinite(surface)]
    return float(np.median(values)) if values.size else float("nan")


def project_trace(
    values: np.ndarray,
    depth_min: float,
    depth_max: float,
    min_gap: float,
) -> np.ndarray:
    """Проекция трассы на ограниченное множество упорядоченных глубин.

    После замены z[i] = depth[i] - i*min_gap условие
    depth[i+1] - depth[i] >= min_gap превращается в z[i] <= z[i+1].
    PAVA объединяет соседние нарушающие порядок блоки. Для объединённого
    блока берётся среднее исходных z и оно ограничивается пересечением
    допустимых интервалов глубины.
    """
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    if n == 0:
        return values.copy()

    indices = np.arange(n, dtype=np.float64)
    if depth_max - depth_min < (n - 1) * min_gap:
        raise ValueError(
            "Диапазона глубин недостаточно для размещения всех горизонтов "
            "с заданным минимальным зазором."
        )

    z = values - indices * min_gap
    blocks: list[dict[str, float | int]] = []

    for i in range(n):
        lower = depth_min - i * min_gap
        upper = depth_max - i * min_gap
        blocks.append(
            {
                "start": i,
                "end": i,
                "sum": float(z[i]),
                "count": 1,
                "lower": lower,
                "upper": upper,
                "value": float(np.clip(z[i], lower, upper)),
            }
        )

        while (
            len(blocks) >= 2
            and float(blocks[-2]["value"]) > float(blocks[-1]["value"]) + 1e-12
        ):
            right = blocks.pop()
            left = blocks.pop()
            lower_bound = max(float(left["lower"]), float(right["lower"]))
            upper_bound = min(float(left["upper"]), float(right["upper"]))
            if lower_bound > upper_bound:
                raise ValueError(
                    "Не удалось объединить нарушающие порядок горизонты: "
                    "пересечение допустимых интервалов пусто."
                )

            total_sum = float(left["sum"]) + float(right["sum"])
            total_count = int(left["count"]) + int(right["count"])
            mean = total_sum / total_count
            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "sum": total_sum,
                    "count": total_count,
                    "lower": lower_bound,
                    "upper": upper_bound,
                    "value": float(np.clip(mean, lower_bound, upper_bound)),
                }
            )

    projected_z = np.empty(n, dtype=np.float64)
    for block in blocks:
        start = int(block["start"])
        end = int(block["end"])
        projected_z[start : end + 1] = float(block["value"])

    return projected_z + indices * min_gap


def adjacent_report(horizons: np.ndarray, ids: list[str], min_gap: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for i in range(len(ids) - 1):
        upper = horizons[i]
        lower = horizons[i + 1]
        pair_valid = np.isfinite(upper) & np.isfinite(lower)
        gaps = lower[pair_valid] - upper[pair_valid]
        key = f"{ids[i]} -> {ids[i + 1]}"
        result[key] = {
            "comparable_traces": int(gaps.size),
            "crossings": int(np.count_nonzero(gaps <= 0.0)),
            "gap_below_minimum": int(np.count_nonzero(gaps < min_gap - 1e-6)),
            "minimum_observed_gap": float(np.min(gaps)) if gaps.size else None,
            "p01_gap": float(np.percentile(gaps, 1)) if gaps.size else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work", type=Path, default=DEFAULT_WORK,
        help="Папка с horizons_raw.bin и horizon_ids.txt (обычно rgt_work/mhe_work).",
    )
    parser.add_argument(
        "--meta", type=Path, default=None,
        help="Путь к volume_meta.json; по умолчанию поиск в --work и его родителе.",
    )
    parser.add_argument(
        "--rgt-dir", type=Path, default=None,
        help="Папка результатов RGT/QC; по умолчанию <work>/rgt_ordered.",
    )
    parser.add_argument(
        "--min-gap", type=float, default=DEFAULT_MIN_GAP,
        help="Минимальное расстояние между соседними горизонтами в отсчётах глубины.",
    )
    parser.add_argument(
        "--no-build", action="store_true",
        help="Только исправить и проверить горизонты, не строить RGT.",
    )
    args = parser.parse_args()

    work = args.work.resolve()
    # Поддерживаем оба распространённых варианта: Java output-dir прямо в
    # mhe_work или отдельная папка mhe_work/extracted.
    if (
        args.work == DEFAULT_WORK
        and not (work / "horizons_raw.bin").is_file()
        and (work / "extracted" / "horizons_raw.bin").is_file()
    ):
        work = work / "extracted"
    meta_path = resolve_meta_path(work, args.meta)
    rgt_dir = args.rgt_dir.resolve() if args.rgt_dir else work / "rgt_ordered"
    raw_path = work / "horizons_raw.bin"
    ids_path = work / "horizon_ids.txt"
    ordered_path = work / "horizons_ordered.bin"
    mask_path = work / "horizons_correction_mask.npy"
    report_path = rgt_dir / "horizon_correction_report.json"

    if not work.is_dir():
        raise NotADirectoryError(f"Рабочая папка не существует: {work}")
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Не найден {raw_path}. Проверь output-dir в запуске ExtractHorizons.java: "
            "именно туда Java записывает horizons_raw.bin и horizon_ids.txt."
        )
    if args.min_gap <= 0:
        raise ValueError("--min-gap должен быть положительным числом.")

    ids = read_horizon_ids(ids_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    try:
        n_depth = int(meta["n1_depth"])
        n_inline = int(meta["n2_inline"])
        n_cross = int(meta["n3_crossline"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "В volume_meta.json нужны целочисленные поля "
            "n1_depth, n2_inline и n3_crossline."
        ) from exc

    if min(n_depth, n_inline, n_cross) <= 0:
        raise ValueError(f"Некорректные размеры куба: {(n_cross, n_depth, n_inline)}")
    if n_depth < 3:
        raise ValueError("Для построения RGT глубина куба должна быть не меньше 3.")

    # 04_build_rgt.py добавляет граничные узлы (0 и n_depth-1), поэтому
    # реальные горизонты должны находиться строго внутри: [1, n_depth-2].
    depth_min = 1.0
    depth_max = float(n_depth - 2)

    expected = len(ids) * n_cross * n_inline
    raw = np.fromfile(raw_path, dtype="<f4")
    if raw.size != expected:
        expected_bytes = expected * 4
        actual_bytes = raw_path.stat().st_size
        raise ValueError(
            "Размер horizons_raw.bin не совпадает с размерами из volume_meta.json "
            "и количеством ID.\n"
            f"Получено: {raw.size} float32 ({actual_bytes} байт).\n"
            f"Ожидалось: {expected} float32 ({expected_bytes} байт), "
            f"форма ({len(ids)}, {n_cross}, {n_inline}).\n"
            "Проверь, что Java записала little-endian float32 и что --work указывает "
            "на ту же папку, куда Java записала horizons_raw.bin."
        )

    original = raw.reshape(len(ids), n_cross, n_inline).astype(np.float64)
    finite_values = original[np.isfinite(original)]
    if finite_values.size == 0:
        raise ValueError("В horizons_raw.bin нет ни одного конечного значения.")

    p01, p50, p99 = np.percentile(finite_values, [1, 50, 99])
    print("=== Входные данные Java MHE ===")
    print(f"Рабочая папка: {work}")
    print(f"Meta: {meta_path}")
    print(f"Размеры: horizons={len(ids)}, crossline={n_cross}, inline={n_inline}")
    print(f"Глубина для RGT: [{depth_min:.0f}, {depth_max:.0f}]")
    print(f"Исходные глубины, p01/median/p99: {p01:.3f} / {p50:.3f} / {p99:.3f}")

    # Защита от очевидно неверной интерпретации endian/формата.
    if np.percentile(np.abs(finite_values), 99) < 1e-4:
        raise ValueError(
            "Почти все значения близки к нулю. Вероятна ошибка endian или формата "
            "horizons_raw.bin. В ExtractHorizons.java запись горизонтов должна быть "
            "little-endian float32 через writeLE()."
        )
    if np.percentile(np.abs(finite_values), 99) > max(10000.0, 100.0 * n_depth):
        raise ValueError(
            "Значения горизонтов на порядки больше глубины куба. Вероятна ошибка "
            "формата/порядка байтов в horizons_raw.bin."
        )

    medians = np.array([finite_median(surface) for surface in original], dtype=np.float64)
    if not np.all(np.isfinite(medians)):
        bad_ids = [ids[i] for i, value in enumerate(medians) if not np.isfinite(value)]
        raise ValueError("Полностью пустые поверхности нельзя упорядочить: " + ", ".join(bad_ids))

    # Тот же принцип, что и в 04_build_rgt.py: сверху вниз по медиане глубины.
    order = np.argsort(medians, kind="stable")
    sorted_ids = [ids[int(i)] for i in order]
    sorted_original = original[order].copy()
    sorted_medians = medians[order]

    print("\nПорядок горизонтов сверху вниз (по медиане глубины):")
    for horizon_id, median in zip(sorted_ids, sorted_medians):
        print(f"  {horizon_id}: median={median:.3f}")

    before_adjacent = adjacent_report(sorted_original, sorted_ids, args.min_gap)
    n_horizons = len(ids)
    flat = sorted_original.reshape(n_horizons, -1)
    corrected_flat = flat.copy()
    mask_flat = np.zeros(flat.shape, dtype=bool)
    complete = np.all(np.isfinite(flat), axis=0)
    valid_indices = np.flatnonzero(complete)
    incomplete_count = int(flat.shape[1] - valid_indices.size)

    # Запас 1e-4 отсчёта предотвращает округление gap=1.0 в float32 до 0.99999...
    projection_gap = args.min_gap + FLOAT32_GAP_MARGIN
    if depth_max - depth_min < (n_horizons - 1) * projection_gap:
        raise ValueError(
            f"Нельзя разместить {n_horizons} горизонтов с зазором {projection_gap:g} "
            f"в диапазоне [{depth_min:g}, {depth_max:g}]."
        )

    print("\n=== Коррекция ===")
    print(f"Всего трасс: {flat.shape[1]}")
    print(f"Трасс со всеми конечными горизонтами: {valid_indices.size}")
    print(f"Трасс с NaN/Inf, оставленных без изменения: {incomplete_count}")
    print(f"Целевой зазор при коррекции: {projection_gap:.6f}")

    changed_trace_count = 0
    for number, trace_idx in enumerate(valid_indices, start=1):
        values = flat[:, trace_idx]
        gaps = np.diff(values)
        needs_fix = (
            np.any(gaps < projection_gap - 1e-8)
            or np.any(values < depth_min)
            or np.any(values > depth_max)
        )
        if needs_fix:
            projected = project_trace(values, depth_min, depth_max, projection_gap)
            corrected_flat[:, trace_idx] = projected
            changed = np.abs(projected - values) > 1e-6
            mask_flat[:, trace_idx] = changed
            if np.any(changed):
                changed_trace_count += 1

        if number % 100_000 == 0 or number == valid_indices.size:
            print(f"Обработано полных трасс: {number}/{valid_indices.size}")

    corrected_sorted = corrected_flat.reshape(n_horizons, n_cross, n_inline)
    mask_sorted = mask_flat.reshape(n_horizons, n_cross, n_inline)

    # Возвращаемся к исходному порядку, чтобы файл оставался согласован с
    # исходным horizon_ids.txt. 04_build_rgt.py затем повторит сортировку.
    corrected_input_order = np.empty_like(corrected_sorted)
    mask_input_order = np.empty_like(mask_sorted)
    corrected_input_order[order] = corrected_sorted
    mask_input_order[order] = mask_sorted

    ordered_path.parent.mkdir(parents=True, exist_ok=True)
    corrected_input_order.astype("<f4").tofile(ordered_path)
    np.save(mask_path, mask_input_order, allow_pickle=False)

    # QC считаем по фактически записанному float32-файлу, а не по float64 в памяти.
    stored_raw = np.fromfile(ordered_path, dtype="<f4")
    stored = stored_raw.reshape(len(ids), n_cross, n_inline).astype(np.float64)
    stored_sorted = stored[order]
    after_adjacent = adjacent_report(stored_sorted, sorted_ids, args.min_gap)
    stored_finite = np.isfinite(stored_sorted)
    complete_after = np.all(stored_finite, axis=0)
    if np.any(complete_after):
        after_gaps = np.diff(stored_sorted, axis=0)[:, complete_after]
        if np.any(after_gaps < args.min_gap - 1e-6):
            raise RuntimeError(
                "После сохранения float32 остались нарушения минимального зазора. "
                "RGT не будет построен; смотри horizons_ordered.bin."
            )
        after_complete_values = stored_sorted[:, complete_after]
        if np.any(after_complete_values < depth_min - 1e-6) or np.any(after_complete_values > depth_max + 1e-6):
            raise RuntimeError(
                "После сохранения float32 остались глубины вне допустимых границ. "
                "RGT не будет построен."
            )

    stats: dict[str, Any] = {}
    out_of_bounds_before: dict[str, int] = {}
    out_of_bounds_after: dict[str, int] = {}
    for idx, horizon_id in enumerate(ids):
        before_surface = original[idx]
        after_surface = stored[idx]
        finite_before = np.isfinite(before_surface)
        shifts = np.abs(after_surface[finite_before] - before_surface[finite_before])
        changed = finite_before & (np.abs(after_surface - before_surface) > 1e-6)
        stats[horizon_id] = {
            "finite_samples": int(np.count_nonzero(finite_before)),
            "changed_samples": int(np.count_nonzero(changed)),
            "mean_absolute_shift": float(np.mean(shifts)) if shifts.size else None,
            "p95_absolute_shift": float(np.percentile(shifts, 95)) if shifts.size else None,
            "max_absolute_shift": float(np.max(shifts)) if shifts.size else None,
            "nonfinite_samples_left_unchanged": int(np.count_nonzero(~finite_before)),
        }
        out_of_bounds_before[horizon_id] = int(
            np.count_nonzero(finite_before & ((before_surface < depth_min) | (before_surface > depth_max)))
        )
        finite_after = np.isfinite(after_surface)
        out_of_bounds_after[horizon_id] = int(
            np.count_nonzero(finite_after & ((after_surface < depth_min - 1e-6) | (after_surface > depth_max + 1e-6)))
        )

    report = {
        "method": "bounded isotonic regression (PAVA)",
        "input_file": str(raw_path),
        "output_file": str(ordered_path),
        "input_format": "little-endian float32; [horizon, crossline, inline]",
        "shape": [len(ids), n_cross, n_inline],
        "horizon_ids_in_input_order": ids,
        "horizon_ids_top_to_bottom": sorted_ids,
        "median_depths_top_to_bottom_before_correction": {
            horizon_id: float(median) for horizon_id, median in zip(sorted_ids, sorted_medians)
        },
        "depth_bounds_inclusive": [depth_min, depth_max],
        "required_min_gap": float(args.min_gap),
        "projection_gap_with_float32_safety_margin": float(projection_gap),
        "total_traces": int(flat.shape[1]),
        "complete_finite_traces": int(valid_indices.size),
        "traces_with_nonfinite_horizons_left_unchanged": incomplete_count,
        "traces_requiring_correction": int(changed_trace_count),
        "before_adjacent_horizon_checks": before_adjacent,
        "after_adjacent_horizon_checks_from_saved_float32": after_adjacent,
        "out_of_bounds_samples_before": out_of_bounds_before,
        "out_of_bounds_samples_after": out_of_bounds_after,
        "per_horizon_correction": stats,
        "correction_mask_file": str(mask_path),
        "rgt_built": not args.no_build,
    }
    rgt_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Итог коррекции ===")
    print(f"Трасс, где изменились глубины: {changed_trace_count}")
    for pair, values in after_adjacent.items():
        print(
            f"{pair}: пересечений={values['crossings']}, "
            f"зазоров меньше {args.min_gap:g}={values['gap_below_minimum']}, "
            f"минимальный зазор={values['minimum_observed_gap']}"
        )
    print(f"Исправленные поверхности: {ordered_path}")
    print(f"Маска изменений: {mask_path}")
    print(f"Отчёт: {report_path}")

    if not args.no_build:
        if not BUILD_SCRIPT.is_file():
            raise FileNotFoundError(f"Не найден скрипт построения RGT: {BUILD_SCRIPT}")
        command = [
            sys.executable,
            str(BUILD_SCRIPT),
            "--work", str(work),
            "--meta", str(meta_path),
            "--output", str(rgt_dir / "rgt.npy"),
            "--horizons-file", ordered_path.name,
            "--min-gap", str(args.min_gap),
        ]
        print("\n=== Построение RGT по исправленным горизонтам ===", flush=True)
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)
        print(f"\nRGT сохранён: {rgt_dir / 'rgt.npy'}")

    print("\nГОТОВО")


if __name__ == "__main__":
    main()
