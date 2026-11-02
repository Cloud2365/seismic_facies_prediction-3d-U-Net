
from pathlib import Path
import json

import h5py
import numpy as np


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

MY_DATA_DIR = (
    PROJECT_DIR
    / "pytorch-3dunet-master"
    / "my_data"
)

HORIZONS_DIR = MY_DATA_DIR / "horizons"
ALL_DATA_H5 = MY_DATA_DIR / "all_data.h5"

OUTPUT_DIR = SCRIPT_DIR / "rgt_output"

OUTPUT_FILE = OUTPUT_DIR / "rgt_volume.npy"
TEMP_FILE = OUTPUT_DIR / "rgt_volume.tmp.npy"
METADATA_FILE = OUTPUT_DIR / "rgt_metadata.json"


# ============================================================
# VOLUME PARAMETERS
# ============================================================

N_CROSS = 951
N_DEPTH = 462
N_INLINE = 651

VOLUME_SHAPE = (N_CROSS, N_DEPTH, N_INLINE)

# Vertical coordinate: two-way travel time
FIRST_TIME_MS = 4.0
DT_MS = 4.0

# Geological order: top to bottom
HORIZON_ORDER = [9, 8, 7, 6, 5, 4, 3, 2, 1]

# Minimum gap between corrected surfaces (samples)
MIN_GAP = 0.05

# Tolerance when mapping XY to grid indices
GRID_TOLERANCE = 1e-3


# ============================================================
# VALIDATE PATHS AND HDF5
# ============================================================

def validate_inputs():

    print("=" * 75)
    print("INPUT VALIDATION")
    print("=" * 75)

    for path in [MY_DATA_DIR, HORIZONS_DIR, ALL_DATA_H5]:
        print(f"\n{path}")
        print("Exists:", path.exists())

        if not path.exists():
            raise FileNotFoundError(path)

    with h5py.File(ALL_DATA_H5, "r") as h5:

        if "raw" not in h5:
            raise KeyError("Dataset 'raw' not found")

        if h5["raw"].shape != VOLUME_SHAPE:
            raise ValueError(
                f"Unexpected HDF5 shape: {h5['raw'].shape}"
            )

        print("\nHDF5 raw shape:", h5["raw"].shape)
        print("HDF5 raw dtype:", h5["raw"].dtype)

    for number in range(1, 10):

        path = HORIZONS_DIR / f"H{number}.xyz"

        if not path.exists():
            raise FileNotFoundError(path)

    print("\nAll inputs validated.")


# ============================================================
# BUILD XY GRID FROM H8
# ============================================================

def get_coordinate_transform():

    print("\n" + "=" * 75)
    print("BUILDING COORDINATE TRANSFORM")
    print("=" * 75)

    # H8 has exactly one point per XY grid position.
    h8 = np.loadtxt(HORIZONS_DIR / "H8.xyz")

    expected_points = N_INLINE * N_CROSS

    if len(h8) != expected_points:
        raise ValueError(
            f"H8 points: {len(h8)}, expected: {expected_points}"
        )

    # Verified ordering: inline first, crossline second.
    grid = h8.reshape(N_INLINE, N_CROSS, 3)

    origin = grid[0, 0, :2]

    # Horizontal coordinate step along crossline.
    v_cross = np.median(
        np.diff(grid[:, :, :2], axis=1).reshape(-1, 2),
        axis=0
    )

    # Horizontal coordinate step along inline.
    v_inline = np.median(
        np.diff(grid[:, :, :2], axis=0).reshape(-1, 2),
        axis=0
    )

    # Columns correspond to crossline and inline.
    basis = np.column_stack([v_cross, v_inline])

    inverse_basis = np.linalg.inv(basis)

    print("Origin:", origin)
    print("Crossline vector:", v_cross)
    print("Inline vector:", v_inline)

    print("Coordinate transform ready.")

    return origin, inverse_basis


# ============================================================
# MAP ORIGINAL XYZ HORIZONS TO HDF5 GRID
# ============================================================

def load_horizons():

    origin, inverse_basis = get_coordinate_transform()

    print("\n" + "=" * 75)
    print("MAPPING ORIGINAL XYZ HORIZONS")
    print("=" * 75)

    # Shape: [9 horizons, crossline, inline]
    depths = np.full(
        (9, N_CROSS, N_INLINE),
        np.nan,
        dtype=np.float32
    )

    for level, number in enumerate(HORIZON_ORDER):

        path = HORIZONS_DIR / f"H{number}.xyz"

        points = np.loadtxt(path)

        xy = points[:, :2]
        z = points[:, 2]

        # Convert real coordinates into grid indices.
        indices = (
            inverse_basis @ (xy - origin).T
        ).T

        cross_float = indices[:, 0]
        inline_float = indices[:, 1]

        cross_idx = np.rint(cross_float).astype(np.int32)
        inline_idx = np.rint(inline_float).astype(np.int32)

        # Check horizontal grid alignment.
        cross_error = np.abs(cross_float - cross_idx)
        inline_error = np.abs(inline_float - inline_idx)

        aligned = (
            (cross_error <= GRID_TOLERANCE)
            & (inline_error <= GRID_TOLERANCE)
        )

        inside = (
            (cross_idx >= 0)
            & (cross_idx < N_CROSS)
            & (inline_idx >= 0)
            & (inline_idx < N_INLINE)
        )

        # Convert vertical time to fractional depth index.
        depth_idx = (z - FIRST_TIME_MS) / DT_MS

        depth_inside = (
            (depth_idx >= 0)
            & (depth_idx <= N_DEPTH - 1)
        )

        valid = aligned & inside & depth_inside

        if not np.all(valid):
            raise ValueError(
                f"H{number}: "
                f"{np.count_nonzero(~valid)} points could not be mapped"
            )

        # Check duplicate grid positions.
        flat_idx = cross_idx * N_INLINE + inline_idx

        if np.unique(flat_idx).size != len(points):
            raise ValueError(
                f"H{number}: duplicate XY grid positions"
            )

        depths[level, cross_idx, inline_idx] = (
            depth_idx.astype(np.float32)
        )

        coverage = np.isfinite(depths[level]).mean() * 100

        print(
            f"H{number}: "
            f"{len(points):,} points | "
            f"coverage={coverage:.3f}%"
        )

    print("\nAll horizons mapped.")

    return depths


# ============================================================
# CORRECT LOCAL HORIZON CROSSINGS
# ============================================================

def correct_horizon_order(depths):

    print("\n" + "=" * 75)
    print("CORRECTING LOCAL HORIZON CROSSINGS")
    print("=" * 75)

    correction_counts = {}
    max_displacements = {}

    # The surfaces are ordered from H9 to H1.
    # Increasing index means deeper position.
    #
    # For each trace, if a deeper horizon is above
    # the preceding available horizon, move it just
    # below the preceding surface.
    #
    # The original XYZ files remain unchanged.

    for c in range(N_CROSS):

        for i in range(N_INLINE):

            column = depths[:, c, i]

            valid_idx = np.flatnonzero(
                np.isfinite(column)
            )

            if len(valid_idx) < 2:
                raise ValueError(
                    f"Insufficient horizons at ({c}, {i})"
                )

            for q in range(1, len(valid_idx)):

                upper_level = valid_idx[q - 1]
                lower_level = valid_idx[q]

                upper_depth = column[upper_level]
                lower_depth = column[lower_level]

                if lower_depth <= upper_depth:

                    corrected_depth = (
                        upper_depth + MIN_GAP
                    )

                    displacement = (
                        corrected_depth - lower_depth
                    )

                    key = (
                        f"H{HORIZON_ORDER[upper_level]}_"
                        f"H{HORIZON_ORDER[lower_level]}"
                    )

                    correction_counts[key] = (
                        correction_counts.get(key, 0) + 1
                    )

                    max_displacements[key] = max(
                        max_displacements.get(key, 0.0),
                        float(displacement)
                    )

                    column[lower_level] = corrected_depth

    print("Crossing correction completed.")

    if correction_counts:
        for key, count in correction_counts.items():
            print(
                f"{key}: {count} corrections | "
                f"max shift={max_displacements[key]:.4f} samples"
            )
    else:
        print("No crossing corrections were necessary.")

    return correction_counts, max_displacements


# ============================================================
# BUILD RGT VOLUME
# ============================================================

def build_rgt(depths):

    print("\n" + "=" * 75)
    print("BUILDING RGT VOLUME")
    print("=" * 75)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if TEMP_FILE.exists():
        TEMP_FILE.unlink()

    # Disk-backed array: avoids loading the entire
    # 1.14 GB RGT volume into RAM.
    rgt_volume = np.lib.format.open_memmap(
        TEMP_FILE,
        mode="w+",
        dtype=np.float32,
        shape=VOLUME_SHAPE
    )

    depth_axis = np.arange(
        N_DEPTH,
        dtype=np.float32
    )

    # RGT values corresponding to H9, H8, ..., H1.
    rgt_levels = np.arange(9, dtype=np.float32)

    total_corrections = 0

    # Process one crossline at a time.
    for c in range(N_CROSS):

        section = np.empty(
            (N_DEPTH, N_INLINE),
            dtype=np.float32
        )

        for i in range(N_INLINE):

            control_depths = depths[:, c, i]

            valid = np.isfinite(control_depths)

            z = control_depths[valid].astype(
                np.float64
            )

            t = rgt_levels[valid].astype(
                np.float64
            )

            if len(z) < 2:
                raise ValueError(
                    f"Less than two horizons at ({c}, {i})"
                )

            # z is strictly increasing after correction.
            if np.any(np.diff(z) <= 0):
                raise ValueError(
                    f"Horizon order error at ({c}, {i})"
                )

            # Interpolate RGT between interpreted horizons.
            curve = np.interp(
                depth_axis,
                z,
                t,
                left=t[0],
                right=t[-1]
            )

            # If the deepest available horizon is not H1,
            # linearly extrapolate using the last two
            # available controls until RGT reaches 8.
            if t[-1] < 8.0:

                slope = (
                    (t[-1] - t[-2])
                    / (z[-1] - z[-2])
                )

                below = depth_axis > z[-1]

                curve[below] = (
                    t[-1]
                    + slope * (depth_axis[below] - z[-1])
                )

            # Limit RGT to the assigned stratigraphic range.
            curve = np.clip(curve, 0.0, 8.0)

            section[:, i] = curve.astype(np.float32)

        rgt_volume[c, :, :] = section

        if (c + 1) % 25 == 0:
            print(
                f"Processed crosslines: "
                f"{c + 1}/{N_CROSS}"
            )

    rgt_volume.flush()

    del rgt_volume

    # Move completed temporary volume into final location.
    TEMP_FILE.replace(OUTPUT_FILE)

    print("\nRGT volume saved:")
    print(OUTPUT_FILE)

    return total_corrections


# ============================================================
# FINAL REPORT AND METADATA
# ============================================================

def create_report(correction_counts, max_displacements):

    print("\n" + "=" * 75)
    print("FINAL RGT STATISTICS")
    print("=" * 75)

    rgt = np.load(OUTPUT_FILE, mmap_mode="r")

    minimum = np.inf
    maximum = -np.inf
    total_sum = 0.0
    count = 0

    for c in range(N_CROSS):

        section = np.asarray(rgt[c])

        minimum = min(
            minimum,
            float(section.min())
        )

        maximum = max(
            maximum,
            float(section.max())
        )

        total_sum += float(section.sum())
        count += section.size

    mean = total_sum / count

    print("Shape:", rgt.shape)
    print("Dtype:", rgt.dtype)
    print("Minimum:", minimum)
    print("Maximum:", maximum)
    print("Mean:", mean)
    print("NaN count:", int(np.isnan(rgt).sum()))

    print(
        "File size:",
        f"{OUTPUT_FILE.stat().st_size / (1024 ** 3):.3f} GB"
    )

    metadata = {
        "shape": list(VOLUME_SHAPE),
        "axis_order": "crossline, depth, inline",
        "dtype": "float32",
        "input_horizons": [
            f"H{n}.xyz"
            for n in range(1, 10)
        ],
        "horizon_order_top_to_bottom": HORIZON_ORDER,
        "rgt_mapping": {
            f"H{n}": int(value)
            for n, value in zip(HORIZON_ORDER, range(9))
        },
        "vertical_coordinate": {
            "first_time_ms": FIRST_TIME_MS,
            "step_ms": DT_MS
        },
        "interpolation": "linear",
        "outside_horizon_range": "constant top / linear bottom extrapolation",
        "crossing_corrections": correction_counts,
        "maximum_displacements_samples": max_displacements,
        "statistics": {
            "min": minimum,
            "max": maximum,
            "mean": mean,
            "nan_count": int(np.isnan(rgt).sum())
        }
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\nMetadata saved:")
    print(METADATA_FILE)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("F3 RGT CONSTRUCTION")
    print("=" * 75)

    validate_inputs()

    depths = load_horizons()

    corrections, displacements = (
        correct_horizon_order(depths)
    )

    build_rgt(depths)

    create_report(
        corrections,
        displacements
    )

    print("\n" + "=" * 75)
    print("RGT CONSTRUCTION COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
