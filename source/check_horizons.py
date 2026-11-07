from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = Path("rgt_work")

SEISMIC_PATH = DATA_DIR / "full_seismic.npy"
HORIZONS_PATH = DATA_DIR / "horizons_seismic.npz"

OUTPUT_DIR = DATA_DIR / "qc_plots"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Загружаем сейсмику без полной загрузки в оперативную память.
    seismic = np.load(
        SEISMIC_PATH,
        mmap_mode="r"
    )

    data = np.load(HORIZONS_PATH)
    horizons = data["depth"]

    print("Seismic:", seismic.shape)
    print("Horizons:", horizons.shape)

    if seismic.shape != (901, 255, 601):
        raise ValueError("Неверная форма сейсмики.")

    # Сортировка для проверки порядка поверхностей.
    medians = np.nanmedian(horizons, axis=(1, 2))
    order = np.argsort(medians)
    sorted_horizons = horizons[order]

    print("\n========== СТАТИСТИКА ГОРИЗОНТОВ ==========")

    for j, h in enumerate(sorted_horizons):
        print(
            f"H{j + 1}: "
            f"median={np.nanmedian(h):.2f}, "
            f"min={np.nanmin(h):.2f}, "
            f"max={np.nanmax(h):.2f}, "
            f"coverage={np.isfinite(h).mean() * 100:.2f}%"
        )

    print("\n========== ПРОВЕРКА ПЕРЕСЕЧЕНИЙ ==========")

    for j in range(len(sorted_horizons) - 1):

        difference = (
            sorted_horizons[j + 1] - sorted_horizons[j]
        )

        print(
            f"H{j + 1} -> H{j + 2}: "
            f"пересечения={(difference <= 0).mean() * 100:.2f}%, "
            f"расстояние < 2 образцов={(difference < 2).mean() * 100:.2f}%, "
            f"минимум={np.nanmin(difference):.2f}, "
            f"медиана={np.nanmedian(difference):.2f}"
        )

    # INLINE-разрезы: горизонтальная ось — crossline.
    inline_positions = [50, 100, 200, 300, 400, 500]

    for inline in inline_positions:

        section = np.asarray(
            seismic[:, :, inline]
        )

        vmin, vmax = np.percentile(section, [2, 98])

        fig, ax = plt.subplots(figsize=(15, 6))

        ax.imshow(
            section.T,
            cmap="gray",
            aspect="auto",
            origin="upper",
            vmin=vmin,
            vmax=vmax
        )

        for j, h in enumerate(sorted_horizons):
            ax.plot(
                np.arange(901),
                h[:, inline],
                linewidth=1,
                label=f"H{j + 1}"
            )

        ax.set_title(f"Inline {inline}")
        ax.set_xlabel("Crossline")
        ax.set_ylabel("Depth")
        ax.set_ylim(254, 0)
        ax.legend(loc="upper right")

        fig.tight_layout()

        path = OUTPUT_DIR / f"inline_{inline}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)

        print("Сохранено:", path)

    # CROSSLINE-разрезы: горизонтальная ось — inline.
    crossline_positions = [100, 300, 500, 700, 850]

    for crossline in crossline_positions:

        section = np.asarray(
            seismic[crossline, :, :]
        )

        vmin, vmax = np.percentile(section, [2, 98])

        fig, ax = plt.subplots(figsize=(15, 6))

        ax.imshow(
            section,
            cmap="gray",
            aspect="auto",
            origin="upper",
            vmin=vmin,
            vmax=vmax
        )

        for j, h in enumerate(sorted_horizons):
            ax.plot(
                np.arange(601),
                h[crossline, :],
                linewidth=1,
                label=f"H{j + 1}"
            )

        ax.set_title(f"Crossline {crossline}")
        ax.set_xlabel("Inline")
        ax.set_ylabel("Depth")
        ax.set_ylim(254, 0)
        ax.legend(loc="upper right")

        fig.tight_layout()

        path = OUTPUT_DIR / f"crossline_{crossline}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)

        print("Сохранено:", path)

    print("\nПроверка завершена.")


if __name__ == "__main__":
    main()