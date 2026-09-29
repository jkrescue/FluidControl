#!/usr/bin/env python3
"""Print the resolved PhysicsNeMo FNO architecture and parameter counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from train_tandem_fno import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    with initialize_config_dir(config_dir=str(args.config.parent.resolve()), version_base="1.3"):
        config = compose(config_name=args.config.stem)
    model = build_model(config)
    report = {
        "config": str(args.config),
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "model": OmegaConf.to_container(config.model, resolve=True),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
