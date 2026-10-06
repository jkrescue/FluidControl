"""Project input adapter only; no evaluator math, split access, or approval logic."""
from pathlib import Path
import hashlib
import inspect

FIELDS = ('state', 'mask', 'omega', 'force', 'time')
READER_SHA = 'cafa65d615555e1e4b1d6cb58895983682aae826957765c105142b71e934caa0'

def read_trajectory(path):
    """Read one explicitly supplied file; caller owns authorized manifest selection.

    Return CPU tensors preserving stored dtypes/shapes plus static NumPy x/y.
    No normalization, casting, split enumeration, caching, or device transfer.
    """
    import h5py
    import numpy as np
    import torch
    from physicsnemo.datapipes.readers.hdf5 import HDF5Reader
    source = Path(inspect.getfile(HDF5Reader))
    if hashlib.sha256(source.read_bytes()).hexdigest() != READER_SHA:
        raise ValueError('official HDF5Reader source differs')
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('explicit regular trajectory required')
    with h5py.File(path, 'r') as handle:
        shapes = {key: handle[key].shape for key in FIELDS}
        if any(not shape or shape[0] != shapes['state'][0] for shape in shapes.values()) or shapes['state'][0] == 0:
            raise ValueError('unequal or empty time axis')
        # Installed official reader uses torch.from_numpy per sample, so each
        # field must have an array-valued frame, including omega/time [N,1].
        if any(len(shape) < 2 for shape in shapes.values()):
            raise ValueError('official reader requires array-valued samples')
        static = {key: np.asarray(handle[key][:]) for key in ('x','y')}
    reader = HDF5Reader(path, fields=list(FIELDS))
    rows = {key: [] for key in FIELDS}
    try:
        for index in range(len(reader)):
            sample, _ = reader[index]
            for key in FIELDS:
                rows[key].append(sample[key])
        result = {key: torch.stack(rows[key], dim=0) for key in FIELDS}
    finally:
        reader.close()
    if any(tuple(result[key].shape) != shapes[key] for key in FIELDS):
        raise ValueError('reader changed field shape')
    return result, static
