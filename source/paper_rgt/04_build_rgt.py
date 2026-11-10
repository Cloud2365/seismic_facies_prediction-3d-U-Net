#!/usr/bin/env python3
"""Построить RGT из поверхностей MHE.

Формат поверхностей: little-endian float32, порядок [horizon][crossline][inline].
На каждой полной и геометрически корректной трассе значения RGT заданы
постоянными на горизонтах и линейно интерполируются между ними по глубине.
Невалидные трассы остаются NaN; этот скрипт не заполняет пропуски.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл ID горизонтов: {path}")
    text = path.read_text(encoding="utf-8-sig").strip().replace("\n", "")
    ids = [value.strip() for value in text.split(",") if value.strip()]
    if not ids:
        raise ValueError("horizon_ids.txt пуст")
    if len(set(ids)) != len(ids):
        raise ValueError("В horizon_ids.txt есть повторяющиеся ID")
    return ids


def resolve_meta(work: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Не найден volume_meta.json: {path}")
        return path
    for path in (work / "volume_meta.json", work.parent / "volume_meta.json"):
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Не найден volume_meta.json. Передай путь через --meta или положи файл "
        "в --work/родительскую папку."
    )


def finite_median(surface: np.ndarray) -> float:
    values = surface[np.isfinite(surface)]
    return float(np.median(values)) if values.size else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True,
                        help="Папка с horizons bin и horizon_ids.txt.")
    parser.add_argument("--meta", type=Path, default=None,
                        help="Путь к volume_meta.json; по умолчанию поиск в --work и родителе.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Путь к итоговому rgt.npy.")
    parser.add_argument("--horizons-file", default="horizons_raw.bin",
                        help="Имя бинарного файла поверхностей (по умолчанию horizons_raw.bin).")
    parser.add_argument("--min-gap", type=float, default=1.0,
                        help="Минимальный зазор между горизонтами в отсчётах глубины.")
    args = parser.parse_args()

    work = args.work.resolve()
    meta_path = resolve_meta(work, args.meta)
    horizons_path = work / args.horizons_file
    if not horizons_path.is_file():
        raise FileNotFoundError(f"Не найден файл поверхностей: {horizons_path}")
    if args.min_gap <= 0:
        raise ValueError("--min-gap должен быть положительным числом.")

    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    try:
        n_depth = int(meta["n1_depth"])
        n_inline = int(meta["n2_inline"])
        n_cross = int(meta["n3_crossline"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "В volume_meta.json нужны n1_depth, n2_inline и n3_crossline."
        ) from exc
    if min(n_depth, n_inline, n_cross) <= 0 or n_depth < 3:
        raise ValueError(f"Некорректные размеры тома: {(n_cross, n_depth, n_inline)}")

    ids = read_ids(work / "horizon_ids.txt")
    raw = np.fromfile(horizons_path, dtype="<f4")
    expected = len(ids) * n_cross * n_inline
    if raw.size != expected:
        raise ValueError(
            f"В {horizons_path} найдено {raw.size} float32, ожидалось {expected}; "
            f"форма должна быть ({len(ids)}, {n_cross}, {n_inline})."
        )
    horizons = raw.reshape(len(ids), n_cross, n_inline)

    medians = np.array([finite_median(surface) for surface in horizons], dtype=np.float64)
    if not np.all(np.isfinite(medians)):
        empty_ids = [ids[i] for i, value in enumerate(medians) if not np.isfinite(value)]
        raise ValueError("У этих горизонтов нет конечных значений: " + ", ".join(empty_ids))

    order = np.argsort(medians, kind="stable")
    horizons = horizons[order]
    ids = [ids[int(i)] for i in order]
    medians = medians[order]

    print("Порядок горизонтов сверху вниз по медиане глубины:")
    for horizon_id, median in zip(ids, medians):
        print(f"  {horizon_id}: {median:.2f}")

    finite = np.isfinite(horizons).all(axis=0)
    depth_min = 1.0
    depth_max = float(n_depth - 2)
    inside = ((horizons >= depth_min) & (horizons <= depth_max)).all(axis=0)
    ordered = (np.diff(horizons, axis=0) >= args.min_gap - 1e-6).all(axis=0)
    valid = finite & inside & ordered

    levels = np.arange(1, len(ids) + 1, dtype=np.float64) / float(len(ids) + 1)
    rgt = np.full((n_cross, n_depth, n_inline), np.nan, dtype=np.float32)
    depth = np.arange(n_depth, dtype=np.float64)

    # Каждая подходящая трасса интерполируется независимо по глубине.
    # Поверхности не экстраполируются за границы куба и не переносятся с соседних трасс.
    valid_crosslines, valid_inlines = np.nonzero(valid)
    for c, i in zip(valid_crosslines, valid_inlines):
        z = horizons[:, c, i].astype(np.float64)
        x = np.concatenate(([0.0], z, [float(n_depth - 1)]))
        y = np.concatenate(([0.0], levels, [1.0]))
        rgt[c, :, i] = np.interp(depth, x, y).astype(np.float32)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, rgt)
    valid_path = output.with_name("rgt_valid_trace.npy")
    np.save(valid_path, valid)

    report: dict[str, Any] = {
        "horizon_ids_top_to_bottom": ids,
        "horizon_median_depths_top_to_bottom": medians.tolist(),
        "rgt_levels": levels.tolist(),
        "depth_bounds_for_horizons": [depth_min, depth_max],
        "required_min_gap": float(args.min_gap),
        "valid_trace_fraction": float(valid.mean()),
        "valid_trace_count": int(np.count_nonzero(valid)),
        "total_trace_count": int(valid.size),
        "incomplete_trace_fraction": float((~finite).mean()),
        "crossing_or_too_close_fraction_among_complete": (
            float((finite & ~ordered).sum() / finite.sum()) if finite.any() else None
        ),
        "out_of_bounds_fraction_among_complete": (
            float((finite & ~inside).sum() / finite.sum()) if finite.any() else None
        ),
        "horizons_file": horizons_path.name,
        "horizons_format": "little-endian float32; [horizon, crossline, inline]",
        "method": "MHE surfaces; constant RGT on each horizon and linear interpolation in depth",
        "labels_used": False,
        "missing_traces_filled": False,
    }
    report_path = output.with_name("rgt_qc.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== RGT QC ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nRGT: {output}")
    print(f"Маска валидных трасс: {valid_path}")
    print(f"Отчёт: {report_path}")
    if valid.mean() < 0.9:
        print("Внимание: валидных трасс меньше 90%. Проверь пересечения, NaN/Inf и контрольные точки.")


if __name__ == "__main__":
    main()
