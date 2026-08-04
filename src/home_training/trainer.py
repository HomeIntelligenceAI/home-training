"""HFM-2 training engine.

Mixed precision, gradient accumulation, gradient checkpointing, cosine schedule
with warmup, and resumable checkpoints — the pieces needed to train a 138M
model on a 4 GB consumer GPU.

The model itself lives in home-transformer. This module owns the loop.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from home_transformer import HFM2, HFM2Config

from home_training.data import TokenDataset
from home_training.schedule import cosine_with_warmup

__all__ = ["TrainConfig", "Trainer", "resolve_device", "resolve_dtype"]


@dataclass
class TrainConfig:
    train_bin: Path
    val_bin: Path | None = None
    out_dir: Path = Path("checkpoints")

    batch_size: int = 4
    """Micro-batch actually resident on the device."""

    grad_accum_steps: int = 8
    """Micro-batches per optimiser step; effective batch is the product. The
    knob that decouples the batch the optimiser sees from the batch the GPU
    can physically hold."""

    max_steps: int = 1000
    warmup_steps: int = 100
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    eval_interval: int = 100
    eval_batches: int = 20
    checkpoint_interval: int = 500
    log_interval: int = 10

    gradient_checkpointing: bool = False
    seed: int = 1337
    device: str | None = None
    dtype: str | None = None
    """'bfloat16' | 'float16' | 'float32'. None picks the best the device has."""

    model: dict[str, object] = field(default_factory=dict)
    """Overrides applied to HFM2Config."""

    @property
    def tokens_per_step(self) -> int:
        seq = int(self.model.get("max_seq_len", HFM2Config.max_seq_len))
        return self.batch_size * self.grad_accum_steps * seq


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(requested: str | None, device: torch.device) -> torch.dtype:
    if requested:
        return getattr(torch, requested)
    if device.type != "cuda":
        # float16 on CPU is slower than float32 on most hardware, and autocast
        # buys nothing for this workload there.
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class Trainer:
    """Owns the training loop and nothing else."""

    def __init__(self, config: TrainConfig, model_config: HFM2Config | None = None):
        self.config = config
        self.device = resolve_device(config.device)
        self.dtype = resolve_dtype(config.dtype, self.device)

        torch.manual_seed(config.seed)
        self.generator = torch.Generator().manual_seed(config.seed)

        self.model_config = model_config or HFM2Config(**config.model)  # type: ignore[arg-type]
        self.model = HFM2(self.model_config).to(self.device)
        if config.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        self.train_data = TokenDataset(config.train_bin, self.model_config.max_seq_len)
        self.val_data = (
            TokenDataset(config.val_bin, self.model_config.max_seq_len)
            if config.val_bin
            else None
        )

        # Fail before the first step rather than after seven days.
        self.vocab_check = self.train_data.check_against_vocab(
            self.model_config.vocab_size
        )

        self.optimizer = torch.optim.AdamW(
            self.model.param_groups(config.weight_decay),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            fused=self.device.type == "cuda",
        )
        # GradScaler matters only for float16. bfloat16 keeps fp32's exponent
        # range, so gradients do not underflow and scaling is pure overhead.
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.dtype is torch.float16
        )

        self.step = 0
        self.best_val_loss = float("inf")
        self.out_dir = Path(config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _autocast(self):  # type: ignore[no-untyped-def]
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.dtype is not torch.float32,
        )

    def describe(self) -> str:
        non_emb = self.model.count_params(non_embedding=True)
        return (
            f"HFM-2  {self.model.count_params() / 1e6:.1f}M params "
            f"({non_emb / 1e6:.1f}M non-embedding)\n"
            f"device {self.device}  dtype {str(self.dtype).replace('torch.', '')}\n"
            f"train  {self.train_data.stats}\n"
            f"val    {self.val_data.stats if self.val_data else 'none'}\n"
            f"batch  {self.config.batch_size} x {self.config.grad_accum_steps} accum "
            f"x {self.model_config.max_seq_len} ctx "
            f"= {self.config.tokens_per_step:,} tokens/step\n"
            f"vocab  {self.vocab_check}"
        )

    def train(self) -> dict[str, float]:
        self.model.train()
        started = time.perf_counter()
        running_loss = 0.0

        while self.step < self.config.max_steps:
            lr = cosine_with_warmup(
                self.step,
                base_lr=self.config.lr,
                warmup_steps=self.config.warmup_steps,
                total_steps=self.config.max_steps,
                min_lr_ratio=self.config.min_lr_ratio,
            )
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            for _ in range(self.config.grad_accum_steps):
                x, y = self.train_data.batch(
                    self.config.batch_size, self.device, self.generator
                )
                with self._autocast():
                    _, loss, _ = self.model(x, targets=y)
                    # Divide so accumulated gradients average rather than sum.
                    # Without this the effective learning rate scales with
                    # grad_accum_steps, and raising accumulation diverges a run
                    # that would otherwise have been fine.
                    loss = loss / self.config.grad_accum_steps
                self.scaler.scale(loss).backward()
                accumulated += loss.item()

            if self.config.grad_clip > 0:
                # Unscale first, or the threshold is compared against scaled
                # gradients and the clip silently does nothing.
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += accumulated
            self.step += 1

            if self.step % self.config.log_interval == 0:
                elapsed = time.perf_counter() - started
                mean = running_loss / self.config.log_interval
                tok_s = self.config.tokens_per_step * self.config.log_interval / elapsed
                print(
                    f"step {self.step:>6}  loss {mean:.4f}  "
                    f"ppl {math.exp(min(mean, 20)):>9.1f}  lr {lr:.2e}  "
                    f"{tok_s:,.0f} tok/s",
                    # Unbuffered: stdout block-buffers when redirected to a
                    # file, so a multi-day run would show nothing until it
                    # exited -- indistinguishable from a hang.
                    flush=True,
                )
                running_loss = 0.0
                started = time.perf_counter()

            if self.val_data and self.step % self.config.eval_interval == 0:
                val = self.evaluate()
                print(
                    f"  eval  loss {val:.4f}  ppl {math.exp(min(val, 20)):.1f}",
                    flush=True,
                )
                if val < self.best_val_loss:
                    self.best_val_loss = val
                    self.save_checkpoint("best.pt")
                self.model.train()

            if self.step % self.config.checkpoint_interval == 0:
                self.save_checkpoint("latest.pt")

        self.save_checkpoint("latest.pt")
        return {"step": float(self.step), "best_val_loss": self.best_val_loss}

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_data is None:
            raise ValueError("no validation set configured")
        self.model.eval()
        batches = self.val_data.sequential_batches(
            self.config.batch_size, self.config.eval_batches, self.device
        )
        total = 0.0
        for x, y in batches:
            with self._autocast():
                _, loss, _ = self.model(x, targets=y)
            total += loss.item()
        return total / max(1, len(batches))

    def save_checkpoint(self, name: str) -> Path:
        path = self.out_dir / name
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "step": self.step,
                "best_val_loss": self.best_val_loss,
                "model_config": self.model_config.to_dict(),
                "train_config": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in asdict(self.config).items()
                },
            },
            path,
        )
        (self.out_dir / "model_config.json").write_text(
            json.dumps(self.model_config.to_dict(), indent=2), encoding="utf-8"
        )
        return path

    def load_checkpoint(self, path: Path) -> None:
        """Restore weights, optimiser state, and step count.

        weights_only=False is required because the payload carries config dicts
        alongside tensors. Only load checkpoints you produced yourself.
        """
        blob = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(blob["model"])
        self.optimizer.load_state_dict(blob["optimizer"])
        self.scaler.load_state_dict(blob["scaler"])
        self.step = blob["step"]
        self.best_val_loss = blob.get("best_val_loss", float("inf"))
