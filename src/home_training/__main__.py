"""Command-line entry point: ``python -m home_training --config configs/....yaml``."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from home_transformer import HFM2Config

from home_training.trainer import TrainConfig, Trainer


def load_config(path: Path) -> tuple[TrainConfig, HFM2Config]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    model_section = raw.pop("model", {}) or {}

    for key in ("train_bin", "val_bin", "out_dir"):
        if raw.get(key):
            # Config paths are written relative to the config file, so a run is
            # reproducible from any working directory.
            raw[key] = (path.parent / raw[key]).resolve()

    train_config = TrainConfig(**raw, model=model_section)
    return train_config, HFM2Config(**model_section)


def main() -> None:
    parser = argparse.ArgumentParser(prog="home_training", description="Train HFM-2")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint to resume from. Restores weights, optimiser, and step.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration and exit without training.",
    )
    args = parser.parse_args()

    train_config, model_config = load_config(args.config)
    trainer = Trainer(train_config, model_config)
    print(trainer.describe())

    if args.dry_run:
        return

    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"resumed from {args.resume} at step {trainer.step}")

    result = trainer.train()
    print(f"\ndone. step {int(result['step'])}, best val loss {result['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
