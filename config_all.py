"""Data-loader configuration for the segmentation workflow."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataloder_Pick import DatasetFromFolder


def _paired_names(data_dir):
    input_dir = Path(data_dir) / "x_train"
    label_dir = Path(data_dir) / "y_train"
    if not input_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            "Expected paired data directories {} and {}".format(input_dir, label_dir)
        )

    input_names = {path.name for path in input_dir.glob("*.mat")}
    label_names = {path.name for path in label_dir.glob("*.mat")}
    missing_labels = sorted(input_names - label_names)
    missing_inputs = sorted(label_names - input_names)
    if missing_labels or missing_inputs:
        raise ValueError(
            "Input/label filenames do not match ({} missing labels, {} missing inputs)".format(
                len(missing_labels), len(missing_inputs)
            )
        )
    return sorted(input_names)


def build_loaders(
    data_dir="jm",
    batch_size=4,
    num_workers=4,
    train_count=1000,
    val_count=200,
    test_count=200,
):
    """Build deterministic train, validation, and test data loaders."""
    names = _paired_names(data_dir)
    required = train_count + val_count + test_count
    if len(names) < required:
        raise ValueError(
            "Requested {} samples, but only {} paired files were found in {}".format(
                required, len(names), data_dir
            )
        )

    train_names = names[:train_count]
    val_names = names[train_count : train_count + val_count]
    test_names = names[train_count + val_count : required]

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        DatasetFromFolder(train_names, data_dir), shuffle=True, **loader_options
    )
    val_loader = DataLoader(
        DatasetFromFolder(val_names, data_dir), shuffle=False, **loader_options
    )
    test_loader = DataLoader(
        DatasetFromFolder(test_names, data_dir), shuffle=False, **loader_options
    )
    return train_loader, val_loader, test_loader
