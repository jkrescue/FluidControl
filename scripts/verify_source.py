"""Verify the immutable Zenodo source archives before any preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

EXPECTED = {
    "README.txt": (3692, "3327aa1fe31312da883bfc484dd8ede5"),
    "ExperimentalDataset.zip": (1333197134, "6629a8e110b1682b9361de37d8be4afb"),
    "URANSDataset.zip": (226542913, "567f6c2602da4174df7e278e380c9012"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/zenodo_20794709"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/source_verification.json"))
    args = parser.parse_args()
    results = {}
    good = True
    for name, (expected_size, expected_md5) in EXPECTED.items():
        path = args.root / name
        if not path.is_file():
            results[name] = {"status": "missing"}
            good = False
            continue
        md5 = hashlib.md5(usedforsecurity=False)
        sha256 = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                md5.update(block)
                sha256.update(block)
        entry = {
            "size_bytes": path.stat().st_size,
            "md5": md5.hexdigest(),
            "sha256": sha256.hexdigest(),
            "size_matches": path.stat().st_size == expected_size,
            "md5_matches": md5.hexdigest() == expected_md5,
        }
        if name.endswith(".zip") and entry["md5_matches"]:
            try:
                with ZipFile(path) as archive:
                    entry["members"] = archive.namelist()
                    entry["bad_member"] = archive.testzip()
            except BadZipFile as exc:
                entry["zip_error"] = str(exc)
        entry["status"] = "verified" if entry["size_matches"] and entry["md5_matches"] and not entry.get("bad_member") and not entry.get("zip_error") else "failed"
        good &= entry["status"] == "verified"
        results[name] = entry
        print(f"{name}: {entry['status']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"source": "https://doi.org/10.5281/zenodo.20794709", "files": results}, indent=2), encoding="utf-8")
    if not good:
        raise SystemExit("Source archives are incomplete or failed integrity checks")


if __name__ == "__main__":
    main()
