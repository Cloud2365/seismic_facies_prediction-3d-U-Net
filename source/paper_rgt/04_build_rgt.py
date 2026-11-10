#!/usr/bin/env python3
"""RGT по разделу 2.2 статьи Wu, Liu, Gu, Ali (2026).

Статья не публикует формулу интерполяции. Сделано ровно то, что в ней написано:
каждому горизонту одно постоянное время, между соседними горизонтами значение
меняется по глубине. Пространственная форма уже сидит в поверхностях Wu:
они посчитаны сразу на всём кубе, а не склеены с соседней трассы.

Порядок горизонтов — по медиане глубины, сверху вниз.
Время горизонта k из H: (k+1)/(H+1). Края трассы: 0 и 1.
Если на трассе горизонты пересеклись или вышли из куба, трасса остаётся пустой.
Так видно ошибку точек, а не замазанный результат.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True, help="Папка с horizons_raw.bin и volume_meta.json")
    p.add_argument("--output", type=Path, required=True, help="Куда писать rgt.npy")
    args = p.parse_args()
    meta = json.loads((args.work / "volume_meta.json").read_text(encoding="utf-8"))
    n_depth = int(meta["n1_depth"])
    n_inline = int(meta["n2_inline"])
    n_cross = int(meta["n3_crossline"])
    ids = (args.work / "horizon_ids.txt").read_text(encoding="utf-8").strip().split(",")
    raw = np.fromfile(args.work / "horizons_raw.bin", dtype="<f4")
    expect = len(ids) * n_cross * n_inline
    if raw.size != expect:
        raise SystemExit(f"В horizons_raw.bin {raw.size} чисел, ожидалось {expect}")
    horizons = raw.reshape(len(ids), n_cross, n_inline)
    order = np.argsort(np.median(horizons, axis=(1, 2)))
    horizons = horizons[order]
    ids = [ids[i] for i in order]
    print("Сверху вниз:")
    for name, surface in zip(ids, horizons):
        print(f"  {name}: медиана глубины {np.median(surface):.1f}")

    finite = np.isfinite(horizons).all(axis=0)
    inside = ((horizons >= 1) & (horizons <= n_depth - 2)).all(axis=0)
    ordered = (np.diff(horizons, axis=0) >= 1.0).all(axis=0)
    valid = finite & inside & ordered
    levels = np.arange(1, len(ids) + 1, dtype=np.float64) / float(len(ids) + 1)
    rgt = np.full((n_cross, n_depth, n_inline), np.nan, dtype=np.float32)
    depth = np.arange(n_depth, dtype=np.float64)
    for c, i in zip(*np.nonzero(valid)):
        z = horizons[:, c, i].astype(np.float64)
        x = np.concatenate(([0.0], z, [float(n_depth - 1)]))
        y = np.concatenate(([0.0], levels, [1.0]))
        rgt[c, :, i] = np.interp(depth, x, y).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, rgt)
    np.save(args.output.with_name("rgt_valid_trace.npy"), valid)
    report = {
        "horizon_ids_top_to_bottom": ids,
        "rgt_levels": levels.tolist(),
        "valid_trace_fraction": float(valid.mean()),
        "crossing_or_too_close_fraction": float((finite & ~ordered).mean()),
        "method": "Wu HorizonExtractor3.surfaceUpdateFromSlopesAndCorrelations, parameters from demo3.py; RGT constant on each horizon, linear in depth between horizons",
        "labels_used": False,
    }
    (args.output.with_name("rgt_qc.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if valid.mean() < 0.9:
        print("Пустых трасс много. Смотрите рисунки: какая-то точка стоит не на том отражении.")


if __name__ == "__main__":
    main()
