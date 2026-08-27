"""Dataset utilities for paired image patches and segmentation labels."""

from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


def load_mat_array(path):
    """Load the single user array stored in a MATLAB file.

    MATLAB metadata keys are ignored. A clear error is raised when a file does
    not contain exactly one data array, avoiding reliance on dictionary order.
    """
    path = Path(path)
    variables = {
        key: value
        for key, value in sio.loadmat(path).items()
        if not key.startswith("__") and isinstance(value, np.ndarray)
    }
    if len(variables) != 1:
        raise ValueError(
            "{} must contain exactly one data array; found {}".format(
                path, sorted(variables)
            )
        )
    return np.asarray(next(iter(variables.values())))


class DatasetFromFolder(torch.utils.data.Dataset):
    """Read matching ``x_train`` inputs and ``y_train`` labels."""

    def __init__(
        self,
        names,
        data_dir,
        input_subdir="x_train",
        label_subdir="y_train",
    ):
        self.names = list(names)
        self.data_dir = Path(data_dir)
        self.input_dir = self.data_dir / input_subdir
        self.label_dir = self.data_dir / label_subdir

    def __getitem__(self, index):
        name = self.names[index]
        factor = load_mat_array(self.input_dir / name).astype(np.float32, copy=False)
        label = np.squeeze(load_mat_array(self.label_dir / name)).astype(
            np.float32, copy=False
        )

        if factor.ndim != 3:
            raise ValueError("Input {} must have shape H x W x C".format(name))
        if label.ndim != 2 or label.shape != factor.shape[:2]:
            raise ValueError(
                "Label {} must have shape {}; found {}".format(
                    name, factor.shape[:2], label.shape
                )
            )

        factor = np.nan_to_num(factor, nan=0.0, posinf=1.0, neginf=0.0)
        factor = np.clip(factor, 0.0, 1.0)
        label = np.nan_to_num(label, nan=0.0, posinf=1.0, neginf=0.0)

        mask = (factor[:, :, :1] > 0).astype(np.float32)
        factor = np.transpose(factor, (2, 0, 1))
        label = label[np.newaxis, :, :]
        mask = np.transpose(mask, (2, 0, 1))
        return factor, label, mask

    def __len__(self):
        return len(self.names)
