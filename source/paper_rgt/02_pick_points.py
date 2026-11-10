#!/usr/bin/env python3
"""Поставить контрольные точки на сейсмике. Разметку не открывает.

Одна команда — один горизонт. На каждом разрезе щёлкните левой кнопкой
по одному и тому же отражению (всё время пик или всё время впадина, не смешивать).
Средняя кнопка или клавиша q — закончить разрез и перейти к следующему.
Точки нужны и на инлайнах, и на кросслайнах: программа Wu держит поверхность
за эти точки.

Файл дописывается, старые точки других горизонтов не стираются.
Повторный запуск с тем же именем горизонта заменяет только его точки.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seismic", type=Path, required=True)
    p.add_argument("--horizon", required=True, help="Имя, например H01")
    p.add_argument("--inlines", default="", help="Номера инлайнов через запятую")
    p.add_argument("--crosslines", default="", help="Номера кросслайнов через запятую")
    p.add_argument("--csv", type=Path, required=True)
    args = p.parse_args()
    if not args.inlines and not args.crosslines:
        raise SystemExit("Укажите --inlines и/или --crosslines")
    vol = np.load(args.seismic, mmap_mode="r")
    n_c, n_z, n_i = vol.shape
    picked: list[tuple[float, float, float]] = []

    def collect(section: str, index: int) -> None:
        if section == "inline":
            if not 0 <= index < n_i:
                raise SystemExit(f"inline {index} вне 0..{n_i - 1}")
            # Строка рисунка — глубина, столбец — кросслайн.
            image = np.asarray(vol[:, :, index]).T
            xlabel = "Crossline"
        else:
            if not 0 <= index < n_c:
                raise SystemExit(f"crossline {index} вне 0..{n_c - 1}")
            # vol[c] уже (depth, inline): строка — глубина, столбец — инлайн.
            image = np.asarray(vol[index])
            xlabel = "Inline"
        lo, hi = np.percentile(image, [2, 98])
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(image, cmap="gray", aspect="auto", vmin=lo, vmax=hi, origin="upper")
        ax.set_title(f"{args.horizon}: {section} {index}. Левая кнопка — точка, средняя — дальше")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Depth")
        local: list[tuple[float, float]] = []

        def onclick(event):
            if event.inaxes is not ax or event.xdata is None:
                return
            if event.button == 2:
                plt.close(fig)
                return
            if event.button != 1:
                return
            x = float(event.xdata)
            z = float(event.ydata)
            local.append((x, z))
            ax.plot(x, z, "o", color="lime", markersize=6)
            ax.annotate(f"{z:.0f}", (x, z), color="yellow", fontsize=8)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()
        for x, z in local:
            if section == "inline":
                picked.append((z, x, float(index)))  # depth, crossline, inline
            else:
                picked.append((z, float(index), x))

    for raw in args.inlines.split(","):
        if raw.strip():
            collect("inline", int(raw))
    for raw in args.crosslines.split(","):
        if raw.strip():
            collect("crossline", int(raw))
    if len(picked) < 1:
        raise SystemExit("Точек нет, файл не изменён")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    for ln in load_rows(args.csv):
        if ln.lower().startswith("horizon_id"):
            continue
        if ln.split(",")[0].strip() == args.horizon:
            continue
        kept.append(ln)
    new_lines = ["horizon_id,depth,crossline,inline"]
    new_lines.extend(kept)
    for z, c, i in picked:
        if not (0 <= z < n_z and 0 <= c < n_c and 0 <= i < n_i):
            raise SystemExit(f"Точка вне куба: depth={z}, crossline={c}, inline={i}")
        new_lines.append(f"{args.horizon},{z:.2f},{c:.2f},{i:.2f}")
    args.csv.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"{args.horizon}: записано {len(picked)} точек в {args.csv}")


if __name__ == "__main__":
    main()
