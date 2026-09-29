"""Print shapes, attributes, and force values without modifying source data."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
from scipy.io import loadmat


def show_mat(path: Path) -> None:
    print(path)
    for key, value in loadmat(path).items():
        if not key.startswith("__"):
            print(f"  {key}: shape={value.shape}, values={value.ravel().tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/zenodo_20794709"))
    parser.add_argument("--no-experimental", action="store_true")
    args = parser.parse_args()
    for path in sorted(args.root.glob("*/AerodynamicForces.mat")):
        show_mat(path)
    archive_path = args.root / "ExperimentalDataset.zip"
    if archive_path.is_file() and not args.no_experimental:
        with ZipFile(archive_path) as archive:
            names = archive.namelist()
            print(f"{archive_path}: {len(names)} members")
            for name in names:
                print(" ", name)
            forces = [name for name in names if name.endswith("AerodynamicForces.mat")]
            if len(forces) == 1:
                print("Experimental force table:")
                for key, value in loadmat(BytesIO(archive.read(forces[0]))).items():
                    if not key.startswith("__"):
                        print(f"  {key}: shape={value.shape}, values={value.ravel().tolist()}")
            examples = [name for name in names if name.endswith(".h5")]
            if examples:
                sample_name = examples[0]
                output = args.root / "inspection_sample" / Path(sample_name).name
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(sample_name) as source, output.open("wb") as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
                with h5py.File(output) as h5:
                    print(f"Sample {sample_name}:")
                    for key, val in h5.attrs.items():
                        print(f"  attr {key}: {val}")
                    h5.visititems(lambda key, val: print(f"  {key}: shape={getattr(val, 'shape', None)}, dtype={getattr(val, 'dtype', None)}"))
                    x, y = np.asarray(h5["X"]), np.asarray(h5["Y"])
                    print(f"  X range: {np.nanmin(x)}, {np.nanmax(x)}; axis changes: {np.nanmean(np.abs(np.diff(x, axis=0)))}, {np.nanmean(np.abs(np.diff(x, axis=1)))}")
                    print(f"  Y range: {np.nanmin(y)}, {np.nanmax(y)}; axis changes: {np.nanmean(np.abs(np.diff(y, axis=0)))}, {np.nanmean(np.abs(np.diff(y, axis=1)))}")
                    sample_u = np.asarray(h5["U"][0])
                    print(f"  U first-frame finite fraction/range: {np.isfinite(sample_u).mean()}, {np.nanmin(sample_u)}, {np.nanmax(sample_u)}")


if __name__ == "__main__":
    main()
