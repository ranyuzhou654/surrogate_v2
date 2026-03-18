"""HDF5 save/load and YAML config utilities."""

import os
from pathlib import Path

import h5py
import numpy as np
import yaml


def save_results(filepath, results):
    """Save experiment results to HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Output path (creates parent directories).
    results : dict
        Nested dict of arrays/scalars to store.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "w") as f:
        _write_group(f, results)


def _write_group(group, d):
    for key, val in d.items():
        if isinstance(val, dict):
            sub = group.create_group(key)
            _write_group(sub, val)
        elif isinstance(val, np.ndarray):
            group.create_dataset(key, data=val)
        elif isinstance(val, (list, tuple)):
            group.create_dataset(key, data=np.array(val))
        elif isinstance(val, (int, float, np.integer, np.floating)):
            group.attrs[key] = val
        elif isinstance(val, str):
            group.attrs[key] = val
        elif val is None:
            group.attrs[key] = "None"
        else:
            group.attrs[key] = str(val)


def load_results(filepath):
    """Load experiment results from HDF5 file.

    Returns
    -------
    dict
        Nested dict mirroring the saved structure.
    """
    results = {}
    with h5py.File(filepath, "r") as f:
        _read_group(f, results)
    return results


def _read_group(group, d):
    for key in group.attrs:
        val = group.attrs[key]
        if val == "None":
            d[key] = None
        else:
            d[key] = val

    for key in group:
        item = group[key]
        if isinstance(item, h5py.Group):
            d[key] = {}
            _read_group(item, d[key])
        else:
            d[key] = item[()]


def load_config(filepath):
    """Load YAML configuration file.

    Parameters
    ----------
    filepath : str or Path
        Path to YAML config.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(filepath, "r") as f:
        return yaml.safe_load(f)
