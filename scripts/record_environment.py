"""Write the isolated runtime and project revision used for a reported run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

PACKAGES = ("nvidia-physicsnemo", "torch", "numpy", "scipy", "h5py", "matplotlib", "ruff")


def command_result(*args: str) -> str | None:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/run_environment.json"))
    args = parser.parse_args()
    package_versions = {}
    for name in PACKAGES:
        try:
            package_versions[name] = version(name)
        except PackageNotFoundError:
            package_versions[name] = None
    report = {
        "utc_time": datetime.now(UTC).isoformat(),
        "git_commit": command_result("git", "rev-parse", "HEAD"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_runtime": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "visible_gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "nvidia_driver": command_result("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
