#!/usr/bin/env python3
"""Переложить сейсмику в порядок, который ждёт программа Wu.

Вход:  full_seismic.npy, оси (crossline, depth, inline)
Выход: seismic_mhe.bin, порядок [crossline][inline][depth], depth самый быстрый,
       big-endian float32. Так устроен массив f[n3][n2][n1] в HorizonExtractor3:
       n1 = depth, n2 = inline, n3 = crossline.
       Это соответствие взято из комментариев demo3.py: p2 — наклон по inline,
       p3 — наклон по crossline.

Метки не читаются.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seismic", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    vol = np.load(args.seismic, mmap_mode="r")
    if vol.ndim != 3:
        raise SystemExit(f"Ожидался объём (crossline, depth, inline), пришло {vol.shape}")
    n_cross, n_depth, n_inline = (int(v) for v in vol.shape)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "seismic_mhe.bin"
    # Пишем по кросслайнам, чтобы не держать в памяти весь переставленный куб.
    with out.open("wb") as f:
        for c in range(n_cross):
            # (depth, inline) -> (inline, depth), как требует цикл i3, i2, i1
            block = np.ascontiguousarray(np.transpose(vol[c], (1, 0)), dtype=">f4")
            block.tofile(f)
            if c % 100 == 0 or c + 1 == n_cross:
                print(f"записан crossline {c + 1}/{n_cross}")
    meta = {
        "n1_depth": n_depth,
        "n2_inline": n_inline,
        "n3_crossline": n_cross,
        "source": str(args.seismic),
        "axis_in_npy": "crossline, depth, inline",
        "mhe_layout": "f[crossline][inline][depth]",
        "labels_read": False,
    }
    (args.output_dir / "volume_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
