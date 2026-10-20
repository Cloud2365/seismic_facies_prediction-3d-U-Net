from abc import abstractmethod
from collections.abc import Iterable
from itertools import chain
from pathlib import Path

import h5py
import numpy as np
import torch

import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.datasets.utils import ConfigDataset, RandomScaler, calculate_stats, get_slice_builder, mirror_pad
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger("HDF5Dataset")


def _create_padded_indexes(indexes: tuple, halo_shape: tuple):
    """Create padded indexes by extending each slice in `indexes` by the corresponding `halo_shape`."""
    if sum(halo_shape) == 0:
        return indexes
    return tuple(slice(index.start, index.stop + 2 * halo) for index, halo in zip(indexes, halo_shape, strict=True))


def traverse_h5_paths(file_paths: list[str]) -> list[str]:
    """Traverse the given list of file paths and directories to find all H5 files."""
    assert isinstance(file_paths, list)
    results = []
    for file_path in file_paths:
        file_path = Path(file_path)
        if file_path.is_dir():
            # if file_path is a directory take all H5 files in that directory
            iters = [file_path.glob(ext) for ext in ["*.h5", "*.hdf", "*.hdf5", "*.hd5"]]
            for fp in chain(*iters):
                results.append(str(fp))
        else:
            results.append(str(file_path))
    return results


class AbstractHDF5Dataset(ConfigDataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride. It's an abstract class for the standard and lazy implementations.

    Args:
        file_path (str): path to H5 file containing raw data and (optional) ground truth labels
        phase (str): 'train' for training, 'val' for validation, 'test' for testing
        slice_builder_config (dict): configuration of the SliceBuilder
        transformer_config (dict): data augmentation configuration
        raw_internal_path (str): H5 internal path to the raw dataset
        label_internal_path (str): H5 internal path to the label dataset
        global_normalization (bool): if True, the mean and std of the raw data will be calculated over the whole dataset
        random_scale (int): if not None, the raw data will be randomly shifted by a value in the range
            [-random_scale, random_scale] in each dimension and then scaled to the original patch shape
        random_scale_probability (float): probability of executing the random scale on a patch
    """

    def __init__(
        self,
        file_path: str,
        phase: str,
        slice_builder_config: dict,
        transformer_config: dict,
        raw_internal_path: str = "raw",
        label_internal_path: str = "label",
        global_normalization: bool = False,
        random_scale: int | None = None,
        random_scale_probability: float = 0.5,
    ):
        assert phase in ["train", "val", "test"]
        logger.info(f"Creating {self.__class__.__name__} for {phase} phase from {file_path}")
        self.phase = phase
        self.file_path = file_path
        self.raw_internal_path = raw_internal_path
        self.label_internal_path = label_internal_path

        self.halo_shape = tuple(slice_builder_config.get("halo_shape", [0, 0, 0]))

        if global_normalization:
            logger.info("Calculating mean and std of the raw data...")
            with h5py.File(file_path, "r") as f:
                raw = f[raw_internal_path][:]
                stats = calculate_stats(raw)
        else:
            stats = calculate_stats(None, True)

        self.transformer = transforms.Transformer(transformer_config, stats)
        self.raw_transform = self.transformer.raw_transform()

        if phase != "test":
            # create label transform only in train/val phase
            self.label_transform = self.transformer.label_transform()
        else:
            # 'test' phase used only for predictions so ignore the label dataset
            self.label = None

            if self.halo_shape == (0, 0, 0):
                logger.warning(
                    "Found halo shape to be (0, 0, 0). This might lead to checkerboard artifacts in the "
                    "prediction. Consider using a non-zero halo shape, e.g. 'halo_shape: [8, 8, 8]' in "
                    "the slice_builder configuration."
                )

        with h5py.File(file_path, "r") as f:
            raw = f[raw_internal_path]
            if raw.ndim == 3:
                self.volume_shape = raw.shape
            else:
                self.volume_shape = raw.shape[1:]
            label = f[label_internal_path] if phase != "test" else None
            # check that raw and label shapes match
            if label is not None:
                if label.ndim == 3:
                    assert label.shape == self.volume_shape, "Raw and label shapes do not match"
                else:
                    assert label.shape[1:] == self.volume_shape, "Raw and label shapes do not match"

            logger.info(f"Volume shape: {self.volume_shape}. Creating slices...")
            # build slice indices for raw and label data sets
            slice_builder_config["lazy_loader"] = self.is_lazy()  # pass this information to the slice builder
            slice_builder = get_slice_builder(raw, label, slice_builder_config)
            self.raw_slices = slice_builder.raw_slices
            self.label_slices = slice_builder.label_slices

        if random_scale is not None:
            assert isinstance(random_scale, int), "random_scale must be an integer"
            stride_shape = slice_builder_config.get("stride_shape")
            assert all(random_scale < stride for stride in stride_shape), (
                f"random_scale {random_scale} must be smaller than each of the strides {stride_shape}"
            )
            patch_shape = slice_builder_config.get("patch_shape")
            self.random_scaler = RandomScaler(random_scale, patch_shape, self.volume_shape, random_scale_probability)
            logger.info(f"Using RandomScaler with offset range {random_scale}")
        else:
            self.random_scaler = None

        self.patch_count = len(self.raw_slices)

    @abstractmethod
    def get_raw_patch(self, idx: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_label_patch(self, idx) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_raw_padded_patch(self, idx) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def is_lazy(self) -> bool:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple:
        if idx >= len(self):
            raise StopIteration

        raw_idx = self.raw_slices[idx]

        if self.phase == "test":
            if len(raw_idx) == 4:
                # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
                raw_idx = raw_idx[1:]  # Remove the first element if raw_idx has 4 elements
                raw_idx_padded = (slice(None),) + _create_padded_indexes(raw_idx, self.halo_shape)
            else:
                raw_idx_padded = _create_padded_indexes(raw_idx, self.halo_shape)

            # use halo padding in the 'test' phase to avoid edge artifacts
            padded_patch = self.get_raw_padded_patch(raw_idx_padded)
            raw_patch_transformed = self.raw_transform(padded_patch)
            # return padded patch, and the original (non-padded) index for placing the prediction back into the volume
            # predictor is responsible for removing the halo from the prediction
            return raw_patch_transformed, raw_idx
        else:
            label_idx = self.label_slices[idx]

            if self.random_scaler is not None:
                # randomize the indices
                raw_idx, label_idx = self.random_scaler.randomize_indices(raw_idx, label_idx)

            raw_patch_transformed = self.raw_transform(self.get_raw_patch(raw_idx))
            label_patch_transformed = self.label_transform(self.get_label_patch(label_idx))

            if self.random_scaler is not None:
                # scale patches back to the original patch size
                raw_patch_transformed, label_patch_transformed = self.random_scaler.rescale_patches(
                    raw_patch_transformed, label_patch_transformed
                )
            # return the transformed raw and label patches
            return raw_patch_transformed, label_patch_transformed

    def __len__(self) -> int:
        return self.patch_count

    @classmethod
    def create_datasets(cls, dataset_config: dict, phase: str) -> Iterable["AbstractHDF5Dataset"]:
        phase_config = dataset_config[phase]

        # load data augmentation configuration
        transformer_config = phase_config["transformer"]
        # load slice builder config
        slice_builder_config = phase_config["slice_builder"]
        # load files to process
        file_paths = phase_config["file_paths"]
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths
        file_paths = traverse_h5_paths(file_paths)

        # create dataset for each file path
        for file_path in file_paths:
            yield cls(
                file_path=file_path,
                phase=phase,
                slice_builder_config=slice_builder_config,
                transformer_config=transformer_config,
                raw_internal_path=dataset_config.get("raw_internal_path", "raw"),
                label_internal_path=dataset_config.get("label_internal_path", "label"),
                global_normalization=dataset_config.get("global_normalization", False),
                random_scale=dataset_config.get("random_scale", None),
                random_scale_probability=dataset_config.get("random_scale_probability", 0.5),
            )


class StandardHDF5Dataset(AbstractHDF5Dataset):
    """Implementation of the HDF5 dataset which loads the data from the H5 files into the memory.
    Fast but might consume a lot of memory.
    """

    def __init__(
        self,
        file_path: str,
        phase: str,
        slice_builder_config: dict,
        transformer_config: dict,
        raw_internal_path: str = "raw",
        label_internal_path: str = "label",
        global_normalization: bool = False,
        random_scale: int = None,
        random_scale_probability: float = 0.5,
    ):
        super().__init__(
            file_path=file_path,
            phase=phase,
            slice_builder_config=slice_builder_config,
            transformer_config=transformer_config,
            raw_internal_path=raw_internal_path,
            label_internal_path=label_internal_path,
            global_normalization=global_normalization,
            random_scale=random_scale,
            random_scale_probability=random_scale_probability,
        )
        self._raw = None
        self._raw_padded = None
        self._label = None

    def get_raw_patch(self, idx):
        # load lazily on the first request
        if self._raw is None:
            with h5py.File(self.file_path, "r") as f:
                assert self.raw_internal_path in f, f"Dataset {self.raw_internal_path} not found in {self.file_path}"
                self._raw = f[self.raw_internal_path][:]
        return self._raw[idx]

    def get_label_patch(self, idx):
        if self._label is None:
            with h5py.File(self.file_path, "r") as f:
                assert self.label_internal_path in f, (
                    f"Dataset {self.label_internal_path} not found in {self.file_path}"
                )
                self._label = f[self.label_internal_path][:]
        return self._label[idx]

    def get_raw_padded_patch(self, idx):
        if self._raw_padded is None:
            with h5py.File(self.file_path, "r") as f:
                assert self.raw_internal_path in f, f"Dataset {self.raw_internal_path} not found in {self.file_path}"
                self._raw_padded = mirror_pad(f[self.raw_internal_path][:], self.halo_shape)
        return self._raw_padded[idx]

    def is_lazy(self) -> bool:
        return False


class LazyHDF5Dataset(AbstractHDF5Dataset):
    """Implementation of the HDF5 dataset which loads the data lazily.
    It's slower, but has a low memory footprint.
    """

    def __init__(
        self,
        file_path: str,
        phase: str,
        slice_builder_config: dict,
        transformer_config: dict,
        raw_internal_path: str = "raw",
        label_internal_path: str = "label",
        global_normalization: bool = False,
        random_scale: int = None,
        random_scale_probability: float = 0.5,
    ):
        super().__init__(
            file_path=file_path,
            phase=phase,
            slice_builder_config=slice_builder_config,
            transformer_config=transformer_config,
            raw_internal_path=raw_internal_path,
            label_internal_path=label_internal_path,
            global_normalization=global_normalization,
            random_scale=random_scale,
            random_scale_probability=random_scale_probability,
        )

        logger.info("Using LazyHDF5Dataset")

    def get_raw_patch(self, idx: int) -> np.ndarray:
        with h5py.File(self.file_path, "r") as f:
            return f[self.raw_internal_path][idx]

    def get_label_patch(self, idx: int) -> np.ndarray:
        with h5py.File(self.file_path, "r") as f:
            return f[self.label_internal_path][idx]

    def get_raw_padded_patch(self, idx: int) -> np.ndarray:
        with h5py.File(self.file_path, "r+") as f:
            if "raw_padded" in f:
                return f["raw_padded"][idx]

            logger.info(f"Creating 'raw_padded' dataset in {self.file_path}")
            raw = f[self.raw_internal_path][:]
            raw_padded = mirror_pad(raw, self.halo_shape)
            f.create_dataset("raw_padded", data=raw_padded, compression="gzip")
            return raw_padded[idx]

    def is_lazy(self) -> bool:
        return True


class LazyHDF5DatasetWithDepth(LazyHDF5Dataset):
    """Lazy HDF5 dataset with an additional absolute-depth channel.

    Output raw patch: ``[2, D, H, W]``
        channel 0: seismic, transformed by the regular ``raw`` transformer from the config
                   (Normalize, augmentations, ToTensor with ``expand_dims: true``);
        channel 1: absolute depth of every voxel, linear in [-1, 1] over the full volume depth.

    The depth channel is appended *after* the transforms. This matters for two reasons:
      * amplitude normalization (global min/max from the config) is applied to seismic only;
      * geometric augmentations (rotation) move the seismic content but must not move the
        depth coordinate — the depth of a voxel position inside the patch is fixed.

    Backward compatibility: configs created before this change had no ``Normalize`` in the raw
    transformer of this dataset (normalization was done per patch inside the dataset). If no
    ``Normalize``/``Standardize`` is present in the raw transformer config, the old per-patch
    min-max normalization is applied so that old checkpoints are evaluated exactly as trained.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raw_cfg = (kwargs.get("transformer_config") or {}).get("raw", [])
        names = {t.get("name") for t in raw_cfg}
        self._legacy_patch_norm = not ({"Normalize", "Standardize", "PercentileNormalizer"} & names)
        if self._legacy_patch_norm:
            logger.warning(
                "LazyHDF5DatasetWithDepth: no normalization in the raw transformer, "
                "falling back to legacy per-patch min-max normalization"
            )
        logger.info("Using LazyHDF5DatasetWithDepth")

    @staticmethod
    def _legacy_normalize(patch: np.ndarray) -> np.ndarray:
        patch = np.asarray(patch, dtype=np.float32)
        lo, hi = float(np.min(patch)), float(np.max(patch))
        return np.clip(2.0 * (patch - lo) / (hi - lo + 1e-10) - 1.0, -1.0, 1.0)

    def _depth_channel(self, raw_idx, spatial_shape) -> np.ndarray:
        if len(raw_idx) == 4:
            raw_idx = raw_idx[1:]
        y = raw_idx[1]
        total_depth = self.volume_shape[1]
        depth = np.linspace(-1.0, 1.0, total_depth, dtype=np.float32)[y.start:y.stop]
        return np.broadcast_to(depth[None, :, None], spatial_shape).astype(np.float32)

    def get_raw_patch(self, idx):
        with h5py.File(self.file_path, "r") as f:
            patch = f[self.raw_internal_path][idx]
        return self._legacy_normalize(patch) if self._legacy_patch_norm else patch

    def get_raw_padded_patch(self, idx):
        # Halo is not used in this project. Read-only access: never write `raw_padded`
        # into the test file (the parent implementation opens it with "r+").
        if any(h != 0 for h in self.halo_shape):
            raise ValueError("LazyHDF5DatasetWithDepth supports only halo_shape = 0")
        return self.get_raw_patch(idx)

    def _append_depth(self, raw_t, raw_idx):
        raw_t = torch.as_tensor(raw_t)
        if raw_t.ndim == 3:
            raw_t = raw_t.unsqueeze(0)
        if raw_t.shape[0] != 1:
            raise ValueError(
                "LazyHDF5DatasetWithDepth expects a single seismic channel after transforms; "
                f"got shape {tuple(raw_t.shape)}. Use ToTensor with expand_dims: true."
            )
        depth = torch.from_numpy(self._depth_channel(raw_idx, tuple(raw_t.shape[1:])))
        return torch.cat([raw_t.float(), depth.unsqueeze(0)], dim=0)

    def __getitem__(self, idx: int) -> tuple:
        if idx >= len(self):
            raise StopIteration
        if self.random_scaler is not None:
            raise ValueError("random_scale is not supported together with the depth channel")
        raw_idx = self.raw_slices[idx]
        out = super().__getitem__(idx)
        return (self._append_depth(out[0], raw_idx),) + tuple(out[1:])
