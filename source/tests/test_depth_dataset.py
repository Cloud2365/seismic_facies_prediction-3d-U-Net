import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")
from pytorch3dunet.datasets.hdf5 import LazyHDF5DatasetWithDepth  # noqa: E402

SB = {"name": "SliceBuilder", "ndim": 3, "patch_shape": [4, 64, 64], "stride_shape": [4, 32, 32]}


@pytest.fixture
def h5file(tmp_path):
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(8, 128, 96)).astype(np.float32)
    label = np.repeat(np.arange(4), 32)[None, :, None].repeat(8, 0).repeat(96, 2)
    path = tmp_path / "t.h5"
    with h5py.File(path, "w") as f:
        f["raw"] = raw
        f["label"] = label
    return str(path), raw


def _ds(path, raw_cfg, phase="train", label_cfg=None):
    return LazyHDF5DatasetWithDepth(
        file_path=path, phase=phase, slice_builder_config=SB,
        transformer_config={"raw": raw_cfg,
                            "label": label_cfg or [{"name": "ToTensor", "expand_dims": False, "dtype": "long"}]},
    )


def test_global_normalization_and_depth_channel(h5file):
    path, raw = h5file
    cfg = [{"name": "Normalize", "min_value": -2.0, "max_value": 2.0}, {"name": "ToTensor", "expand_dims": True}]
    ds = _ds(path, cfg)
    x, y = ds[0]
    assert x.shape == (2, 4, 64, 64)
    sl = ds.raw_slices[0]
    expected = np.clip((raw[sl] + 2.0) / 4.0 * 2 - 1, -1, 1)
    np.testing.assert_allclose(x[0].numpy(), expected, atol=1e-6)
    depth = np.linspace(-1, 1, 128, dtype=np.float32)[sl[1]]
    np.testing.assert_allclose(x[1, 0, :, 0].numpy(), depth)


def test_rotation_does_not_move_depth_channel(h5file):
    path, _ = h5file
    cfg = [{"name": "Normalize", "min_value": -2.0, "max_value": 2.0},
           {"name": "RandomRotate", "angle_spectrum": 30, "order": 1, "axes": [[1, 2]]},
           {"name": "ToTensor", "expand_dims": True}]
    lab = [{"name": "RandomRotate", "angle_spectrum": 30, "order": 0, "axes": [[1, 2]]},
           {"name": "ToTensor", "expand_dims": False, "dtype": "long"}]
    ds = _ds(path, cfg, label_cfg=lab)
    x, _ = ds[1]
    sl = ds.raw_slices[1]
    depth = np.linspace(-1, 1, 128, dtype=np.float32)[sl[1]]
    np.testing.assert_allclose(x[1].numpy(), np.broadcast_to(depth[None, :, None], x[1].shape))


def test_legacy_config_keeps_per_patch_normalization(h5file):
    path, raw = h5file
    ds = _ds(path, [{"name": "ToTensor", "expand_dims": True}], phase="test")
    x, idx = ds[0]
    patch = raw[ds.raw_slices[0]]
    expected = 2 * (patch - patch.min()) / (patch.max() - patch.min() + 1e-10) - 1
    np.testing.assert_allclose(x[0].numpy(), expected, atol=1e-5)
    with h5py.File(path, "r") as f:
        assert "raw_padded" not in f
