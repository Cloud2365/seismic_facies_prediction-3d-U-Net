"""Ключевой эксперимент: сколько информации о разметке несёт RGT-канал.

1) Расстояние между горизонтом (узел RGT) и границей классов в разметке.
2) Декодер, который работает ТОЛЬКО по RGT (без сейсмики): какие метрики он даёт
   на train / test1 / test2. Это нижняя граница того, что может выучить сеть
   с RGT-каналом на входе.
"""
import numpy as np
from pathlib import Path

RAW = Path("../pytorch-3dunet-master/my_data/alaudah_raw")
HOR = RAW / "Horizons"

N_CROSSLINES, N_INLINES, N_DEPTH = 901, 601, 255
FIRST_CROSSLINE, FIRST_INLINE = 300, 100
DEPTH_MIN, DEPTH_MAX = 1005.0, 1877.0
DEPTH_STEP = (DEPTH_MAX - DEPTH_MIN) / (N_DEPTH - 1)

# Группы сверху вниз. "Upper North Sea Group" в файле 58 точек, все вне
# диапазона глубин 1005-1877 м -> в RGT не попадает (см. build_alaudah_rgt.py).
HORIZON_ORDER = [
    "Middle North Sea Group",
    "Lowe North Sea Group",
    "Chalk Group",
    "Rijnland Group",
    "Scruff Group",
    "Zechstein Group",
]
RANKS = {name: i + 1 for i, name in enumerate(HORIZON_ORDER)}  # 1..6 -> RGT = rank/6

LEVELS = np.array([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 6 / 6])
N_CLASSES = 6


def read_surface(name):
    pts = np.loadtxt(HOR / name, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    inline = np.rint(pts[:, 0]).astype(np.int64)
    crossline = np.rint(pts[:, 1]).astype(np.int64)
    z = pts[:, 4]
    i = inline - FIRST_INLINE
    c = crossline - FIRST_CROSSLINE
    ok = ((i >= 0) & (i < N_INLINES) & (c >= 0) & (c < N_CROSSLINES)
          & np.isfinite(z) & (z >= DEPTH_MIN) & (z <= DEPTH_MAX))
    depth_idx = (z[ok] - DEPTH_MIN) / DEPTH_STEP
    grid = np.full((N_CROSSLINES, N_INLINES), np.nan, dtype=np.float32)
    flat = c[ok] * N_INLINES + i[ok]
    sums = np.bincount(flat, weights=depth_idx, minlength=N_CROSSLINES * N_INLINES)
    counts = np.bincount(flat, minlength=N_CROSSLINES * N_INLINES)
    g = grid.reshape(-1)
    m = counts > 0
    g[m] = (sums[m] / counts[m]).astype(np.float32)
    return g.reshape(N_CROSSLINES, N_INLINES)


def build_rgt_trace(hor_depths, hor_ranks):
    """RGT-профиль трассы: кусочно-линейная интерполяция по горизонтам (как в репо)."""
    if len(hor_depths) == 0:
        return None
    order = np.argsort(hor_depths)
    z = hor_depths[order]
    v = hor_ranks[order] / 6.0
    pos = np.concatenate([[0.0], z])
    val = np.concatenate([[0.0], v])
    # фикс пересечений, как в репо
    fixed = np.maximum.accumulate(pos + np.arange(len(pos)) * 0.01)
    return np.interp(np.arange(N_DEPTH, dtype=np.float64), fixed, val,
                     left=val[0], right=val[-1])


def decode_trace(rgt):
    """Классы из одного RGT-профиля: границы = пересечения уровней, класс = по среднему RGT в сегменте."""
    seg_means = []
    bounds = [0]
    for L in LEVELS:
        idx = np.argmax(rgt >= L)
        if rgt[idx] < L:
            break
        bounds.append(idx)
    bounds.append(N_DEPTH)
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            seg_means.append((a, b, float(rgt[a:b].mean())))
    return seg_means


def iou_from_cm(cm):
    tp = np.diag(cm).astype(float)
    sup = cm.sum(axis=1).astype(float)
    pred = cm.sum(axis=0).astype(float)
    union = sup + pred - tp
    iou = np.divide(tp, union, out=np.full(N_CLASSES, np.nan), where=union > 0)
    pa = tp.sum() / cm.sum() if cm.sum() else 0.0
    recall = np.divide(tp, sup, out=np.full(N_CLASSES, np.nan), where=sup > 0)
    return {
        "pixel_accuracy": pa,
        "mean_iou_all": float(np.nanmean(iou)),
        "mean_class_accuracy": float(np.nanmean(recall)),
        "per_class_iou": iou,
    }


def main():
    labels = np.load(RAW / "labels_entire_volume.npy", mmap_mode="r")  # (inline, crossline, depth)
    surfaces = {name: read_surface(name) for name in HORIZON_ORDER}
    for name in HORIZON_ORDER:
        print(f"{name:26s} coverage={np.isfinite(surfaces[name]).mean()*100:6.2f}%")

    # ---------------- Эксперимент A: горизонт vs граница класса ----------------
    print("\n=== A. Расстояние от горизонта до ближайшей границы классов (отсчёты) ===")
    rng = np.random.default_rng(0)
    for name in HORIZON_ORDER:
        surf = surfaces[name]
        cs = rng.choice(N_CROSSLINES, size=120, replace=False)
        dists = []
        for c in cs:
            is_ = np.flatnonzero(np.isfinite(surf[c]))
            if is_.size == 0:
                continue
            sel = is_[rng.choice(is_.size, size=min(60, is_.size), replace=False)]
            for i in sel:
                trace = np.asarray(labels[i, c, :])
                b = np.flatnonzero(np.diff(trace) != 0) + 1
                if b.size == 0:
                    continue
                dists.append(float(np.min(np.abs(b - surf[c, i]))))
        dists = np.asarray(dists)
        if dists.size:
            print(f"{name:26s} n={dists.size:6d} median={np.median(dists):6.2f} "
                  f"p90={np.percentile(dists,90):7.2f} "
                  f"<=1samp={np.mean(dists<=1)*100:5.1f}%  <=3samp={np.mean(dists<=3)*100:5.1f}%")

    # ---------------- Эксперимент B: декодер только по RGT ----------------
    print("\n=== B. Декодер только по RGT (без сейсмики) ===")
    # region -> (inline slice, crossline slice)
    regions = {
        "train": (slice(200, 601), slice(0, 701)),
        "test1": (slice(0, 200), slice(0, 701)),
        "test2": (slice(0, 601), slice(701, 901)),
    }

    # Шаг 1: обучаем отображение "среднее RGT в сегменте" -> класс на TRAIN.
    NBIN = 200
    hist = np.zeros((NBIN, N_CLASSES), dtype=np.int64)
    own_rgt_traces = 0
    total_traces = 0

    for c in range(0, 701, 7):                      # прореживание для скорости обучения
        lab_chunk = np.asarray(labels[0:601, c, :])          # (inline, depth)
        hor_stack = np.stack([surfaces[n][c] for n in HORIZON_ORDER], axis=1)   # (601, 6)
        for i in range(601):
            total_traces += 1
            hd = hor_stack[i][np.isfinite(hor_stack[i])]
            if len(hd) < 2:
                continue
            ranks = np.array([RANKS[n] for n in HORIZON_ORDER])[np.isfinite(hor_stack[i])]
            rgt = build_rgt_trace(hd, ranks)
            own_rgt_traces += 1
            truth = lab_chunk[i]
            for a, b, mean_v in decode_trace(rgt):
                cls = np.bincount(truth[a:b], minlength=N_CLASSES).argmax()
                bin_i = min(NBIN - 1, int(mean_v * NBIN))
                hist[bin_i, cls] += 1

    lookup = hist.argmax(axis=1)
    print(f"Трасc с собственным RGT (train-подмножество): {own_rgt_traces}")
    print("Таблица lookup (среднее RGT в сегменте -> класс), первые 40 бакетов:")
    print("  ", np.round(np.arange(40) / NBIN, 3))
    print("  ", lookup[:40])

    # Шаг 2: применяем декодер на test1 / test2 и считаем метрики.
    results = {}
    for reg in ("test1", "test2"):
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        i_rng, c_rng = regions[reg]
        covered = 0
        for c in range(c_rng.start, c_rng.stop):
            lab_chunk = np.asarray(labels[0:601, c, :])
            hor_stack = np.stack([surfaces[n][c] for n in HORIZON_ORDER], axis=1)
            for i in range(i_rng.start, i_rng.stop):
                hd = hor_stack[i][np.isfinite(hor_stack[i])]
                if len(hd) < 2:
                    continue
                ranks = np.array([RANKS[n] for n in HORIZON_ORDER])[np.isfinite(hor_stack[i])]
                rgt = build_rgt_trace(hd, ranks)
                covered += 1
                pred = np.zeros(N_DEPTH, dtype=np.int64)
                for a, b, mean_v in decode_trace(rgt):
                    bin_i = min(NBIN - 1, int(mean_v * NBIN))
                    pred[a:b] = lookup[bin_i]
                truth = lab_chunk[i]
                cm += np.bincount(N_CLASSES * truth + pred, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
        m = iou_from_cm(cm)
        m["coverage_traces"] = covered
        results[reg] = m
        print(f"\n--- {reg} ---")
        print(f"  трасс с собственным RGT: {covered}")
        print(f"  Pixel Accuracy : {m['pixel_accuracy']:.4f}")
        print(f"  Mean IoU (6 классов): {m['mean_iou_all']:.4f}")
        print(f"  Mean Class Accuracy: {m['mean_class_accuracy']:.4f}")
        print(f"  IoU по классам: {np.round(m['per_class_iou'], 4)}")
        np.savetxt(f"cm_{reg}_rgt_only.csv", cm, delimiter=",")

    print("\nГотово.")


if __name__ == "__main__":
    main()
