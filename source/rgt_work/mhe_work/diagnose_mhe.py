import numpy as np
import csv
from collections import Counter

NPZ = r".\rgt_work\mhe_work\extracted\horizons_mhe.npz"
CSV = r".\rgt_work\mhe_work\control_points.csv"

z = np.load(NPZ)
print("Массивы в NPZ:")
for k in z.files:
    a = z[k]
    print(f"  {k}: shape={a.shape}, dtype={a.dtype}")

candidates = [
    k for k in z.files
    if "horizon" in k.lower()
    and z[k].ndim == 3
    and z[k].shape == (6, 901, 601)
]

if not candidates:
    candidates = [
        k for k in z.files
        if z[k].ndim == 3 and z[k].shape == (6, 901, 601)
    ]

if not candidates:
    raise RuntimeError("Не найден массив горизонтов размера (6,901,601)")

key = candidates[0]
h = z[key].astype(np.float32)
print("\nИспользуем массив:", key)

h05, h06 = h[4], h[5]
valid = np.isfinite(h05) & np.isfinite(h06)
gap = h06 - h05

cross = valid & (gap <= 0)
small_gap = valid & (gap < 1)
print("\n=== H05 -> H06 ===")
print("Сравнимых трасс:", int(valid.sum()))
print("Пересечений (H06 <= H05):", int(cross.sum()))
print("Промежутков меньше 1 отсчёта:", int(small_gap.sum()))

if cross.any():
    coords = np.argwhere(cross)
    order = np.argsort(gap[cross])
    print("\n15 наиболее сильных пересечений:")
    for n in order[:15]:
        c, i = coords[n]
        print(
            f"  Crossline={c}, Inline={i}, "
            f"H05={h05[c,i]:.2f}, H06={h06[c,i]:.2f}, "
            f"gap={gap[c,i]:.2f}"
        )

print("\n=== Подозрительно мелкие значения H06 ===")
for limit in (60, 80, 100, 120):
    mask = np.isfinite(h06) & (h06 < limit)
    print(f"H06 < {limit}: {int(mask.sum())} трасс")

coords = np.argwhere(np.isfinite(h06))
order = np.argsort(h06[np.isfinite(h06)])
print("\n20 самых мелких значений H06:")
for n in order[:20]:
    c, i = coords[n]
    print(f"  Crossline={c}, Inline={i}, H06={h06[c,i]:.2f}")

print("\n=== Участки с наибольшим числом пересечений ===")
bins = Counter((int(c)//100, int(i)//100) for c, i in np.argwhere(cross))
for (c, i), count in bins.most_common(15):
    print(
        f"  Crossline {c*100}:{min((c+1)*100,901)}, "
        f"Inline {i*100}:{min((i+1)*100,601)}: {count}"
    )

print("\n=== Контрольные точки H06 ===")
try:
    with open(CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    print("Колонки CSV:", fields)

    def find_col(names):
        for name in names:
            for field in fields:
                if field.lower().replace("_", "") == name:
                    return field
        return None

    hc = find_col(["horizon", "horizonid"])
    dc = find_col(["depth", "z"])
    cc = find_col(["crossline", "c"])
    ic = find_col(["inline", "i"])

    if hc and dc:
        points = [
            r for r in rows
            if r[hc].strip().upper() == "H06"
        ]
        points.sort(key=lambda r: float(r[dc]))
        print(f"Всего точек H06: {len(points)}")
        print("Точки H06 с наименьшей глубиной:")
        for r in points[:20]:
            print(
                f"  Crossline={r.get(cc) if cc else '?'}, "
                f"Inline={r.get(ic) if ic else '?'}, "
                f"Depth={r[dc]}"
            )
    else:
        print("Не удалось автоматически определить колонки горизонта и глубины.")
except FileNotFoundError:
    print("CSV с контрольными точками не найден:", CSV)

print("\nДиагностика завершена.")
