#!/usr/bin/env python3
"""Наложить готовые горизонты на сейсмику. Смотреть до сборки RGT и до обучения."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seismic", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--inlines", default="300")
    p.add_argument("--crosslines", default="300,500,700")
    args = p.parse_args()
    meta = json.loads((args.work / "volume_meta.json").read_text(encoding="utf-8"))
    n_depth = int(meta["n1_depth"])
    n_inline = int(meta["n2_inline"])
    n_cross = int(meta["n3_crossline"])
    vol = np.load(args.seismic, mmap_mode="r")
    if tuple(vol.shape) != (n_cross, n_depth, n_inline):
        raise SystemExit(f"Куб {vol.shape} не совпал с meta {(n_cross, n_depth, n_inline)}")
    ids = (args.work / "horizon_ids.txt").read_text(encoding="utf-8").strip().split(",")
    horizons = np.fromfile(args.work / "horizons_raw.bin", dtype="<f4").reshape(len(ids), n_cross, n_inline)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def save(image, curves, title, xlabel, path):
        lo, hi = np.percentile(image, [2, 98])
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(image, cmap="gray", aspect="auto", vmin=lo, vmax=hi, origin="upper")
        x = np.arange(curves.shape[1])
        for name, curve in zip(ids, curves):
            ax.plot(x, curve, linewidth=1.1, label=name)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Depth")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    for raw in args.inlines.split(","):
        if not raw.strip():
            continue
        i = int(raw)
        # vol[:, :, i] is (crossline, depth); рисунку нужен (depth, crossline)
        save(np.asarray(vol[:, :, i]).T, horizons[:, :, i], f"Inline {i}", "Crossline",
             args.output_dir / f"inline_{i}.png")
    for raw in args.crosslines.split(","):
        if not raw.strip():
            continue
        c = int(raw)
        save(np.asarray(vol[c]), horizons[:, c, :], f"Crossline {c}", "Inline",
             args.output_dir / f"crossline_{c}.png")
    print(args.output_dir)


if __name__ == "__main__":
    main()
