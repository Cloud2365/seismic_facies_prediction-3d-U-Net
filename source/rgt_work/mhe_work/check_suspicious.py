import numpy as np

z = np.load(r".\rgt_work\mhe_work\extracted\horizons_mhe.npz")
h = z["horizons"]

points = [
    (890, 570),
    (900, 551),
    (900, 553),
    (899, 552),
    (756, 494),
    (755, 494),
    (751, 497),
    (721, 450),
    (619, 550),
    (719, 350),
    (816, 350),
]

print("Crossline | Inline | H05 | H06 | Gap")
print("-" * 55)

for c, i in points:
    a = float(h[4, c, i])
    b = float(h[5, c, i])
    print(f"{c:9d} | {i:6d} | {a:7.2f} | {b:7.2f} | {b-a:8.2f}")

print("\nЛокальный участок около минимального H06:")
for c in range(896, 901):
    for i in range(547, 556):
        a, b = float(h[4,c,i]), float(h[5,c,i])
        print(f"C={c}, I={i}: H05={a:.2f}, H06={b:.2f}, gap={b-a:.2f}")
