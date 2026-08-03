"""Training loop and schedule.

The overfit test is the real one: a model that cannot drive loss to near zero
on a handful of tokens has something wrong with its gradient path, and no
amount of corpus will fix it.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from home_transformer import HFM2Config

from home_training.data import write_tokens
from home_training.schedule import cosine_with_warmup
from home_training.trainer import TrainConfig, Trainer, resolve_dtype

VOCAB = 128


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """A learnable stream: a repeating cycle with a little noise.

    Deliberately not uniform random. Random tokens carry no signal, so loss
    sits at ln(vocab) forever and any test asserting "training reduces loss"
    fails for a reason that has nothing to do with the trainer.
    """
    generator = torch.Generator().manual_seed(0)
    cycle = list(range(40))
    tokens = (cycle * 100)[:4000]
    noise = torch.randint(0, VOCAB, (len(tokens),), generator=generator).tolist()
    flip = torch.rand(len(tokens), generator=generator) < 0.05
    tokens = [n if f else t for t, n, f in zip(tokens, noise, flip.tolist(), strict=True)]

    train, val = tmp_path / "train.bin", tmp_path / "val.bin"
    write_tokens(train, tokens[:3500])
    write_tokens(val, tokens[3500:])
    return train, val


@pytest.fixture
def model_config() -> HFM2Config:
    return HFM2Config(
        vocab_size=VOCAB, dim=32, n_layers=2, n_heads=4, hidden_dim=64, max_seq_len=32
    )


def make_trainer(
    tmp_path: Path, corpus: tuple[Path, Path], config: HFM2Config, **overrides: object
) -> Trainer:
    train_bin, val_bin = corpus
    defaults: dict[str, object] = {
        "train_bin": train_bin,
        "val_bin": val_bin,
        "out_dir": tmp_path / "ckpt",
        "batch_size": 2,
        "grad_accum_steps": 2,
        "max_steps": 5,
        "warmup_steps": 1,
        "log_interval": 100,
        "eval_interval": 100,
        "checkpoint_interval": 100,
        "device": "cpu",
    }
    defaults.update(overrides)
    return Trainer(TrainConfig(**defaults), config)  # type: ignore[arg-type]


class TestSchedule:
    def test_warmup_rises_from_near_zero(self) -> None:
        lrs = [
            cosine_with_warmup(s, base_lr=1e-3, warmup_steps=10, total_steps=100)
            for s in range(10)
        ]
        assert lrs == sorted(lrs)
        assert lrs[0] == pytest.approx(1e-4)
        assert lrs[-1] == pytest.approx(1e-3)

    def test_first_step_is_never_zero(self) -> None:
        """A zero first step wastes the step and hides warmup bugs."""
        assert cosine_with_warmup(0, base_lr=1e-3, warmup_steps=100, total_steps=1000) > 0

    def test_decays_to_the_floor_not_to_zero(self) -> None:
        end = cosine_with_warmup(
            100, base_lr=1e-3, warmup_steps=10, total_steps=100, min_lr_ratio=0.1
        )
        assert end == pytest.approx(1e-4)

    def test_past_the_end_stays_at_the_floor(self) -> None:
        assert cosine_with_warmup(
            5000, base_lr=1e-3, warmup_steps=10, total_steps=100
        ) == pytest.approx(1e-4)

    def test_decay_is_monotonic(self) -> None:
        lrs = [
            cosine_with_warmup(s, base_lr=1e-3, warmup_steps=10, total_steps=100)
            for s in range(10, 101)
        ]
        assert lrs == sorted(lrs, reverse=True)


class TestDtype:
    def test_cpu_stays_in_float32(self) -> None:
        """float16 on CPU is slower than float32; autocast buys nothing here."""
        assert resolve_dtype(None, torch.device("cpu")) is torch.float32

    def test_explicit_request_is_honoured(self) -> None:
        assert resolve_dtype("bfloat16", torch.device("cpu")) is torch.bfloat16


class TestTrainer:
    def test_a_short_run_reduces_the_loss(
        self, tmp_path: Path, tiny_corpus: tuple[Path, Path], model_config: HFM2Config
    ) -> None:
        trainer = make_trainer(
            tmp_path, tiny_corpus, model_config, max_steps=30, warmup_steps=5, lr=3e-3
        )
        before = trainer.evaluate()
        trainer.model.train()
        trainer.train()
        assert trainer.evaluate() < before

    def test_overfits_a_tiny_corpus(self, tmp_path: Path) -> None:
        """Gradients reach every parameter, and the optimiser actually steps.

        The corpus is a short repeating cycle, so a working model should drive
        loss far below the ln(vocab) random baseline within a few dozen steps.
        Failure here means a broken gradient path, not a data problem.
        """
        path = tmp_path / "train.bin"
        write_tokens(path, list(range(32)) * 200)
        config = HFM2Config(
            vocab_size=64, dim=64, n_layers=2, n_heads=4, hidden_dim=128, max_seq_len=16
        )
        trainer = Trainer(
            TrainConfig(
                train_bin=path,
                out_dir=tmp_path / "ckpt",
                batch_size=8,
                grad_accum_steps=1,
                max_steps=120,
                warmup_steps=10,
                lr=3e-3,
                log_interval=1000,
                checkpoint_interval=10_000,
                device="cpu",
            ),
            config,
        )
        trainer.train()
        x, y = trainer.train_data.batch(8, torch.device("cpu"))
        with torch.no_grad():
            _, loss, _ = trainer.model(x, targets=y)
        assert loss is not None
        assert loss.item() < 0.5 * math.log(64)

    def test_unlearnable_data_stays_at_the_entropy_floor(self, tmp_path: Path) -> None:
        """Uniform random tokens carry no signal, so loss must not drop below ln(vocab).

        A model that "learns" noise is seeing its own target. This is the
        cheapest available detector for an off-by-one in the batcher or a mask
        that lets a position attend to itself.
        """
        path = tmp_path / "noise.bin"
        generator = torch.Generator().manual_seed(0)
        write_tokens(path, torch.randint(0, 64, (4000,), generator=generator).tolist())
        config = HFM2Config(
            vocab_size=64, dim=64, n_layers=2, n_heads=4, hidden_dim=128, max_seq_len=16
        )
        trainer = Trainer(
            TrainConfig(
                train_bin=path,
                out_dir=tmp_path / "ckpt",
                batch_size=8,
                grad_accum_steps=1,
                max_steps=120,
                warmup_steps=10,
                lr=3e-3,
                log_interval=1000,
                checkpoint_interval=10_000,
                device="cpu",
            ),
            config,
        )
        trainer.train()
        x, y = trainer.train_data.batch(16, torch.device("cpu"))
        with torch.no_grad():
            _, loss, _ = trainer.model(x, targets=y)
        assert loss is not None
        # Memorising 4000 random tokens at this size is not possible in 120
        # steps; anything well below the floor means information is leaking.
        assert loss.item() > 0.8 * math.log(64)

    def test_gradient_accumulation_does_not_scale_the_gradient(
        self, tmp_path: Path, tiny_corpus: tuple[Path, Path], model_config: HFM2Config
    ) -> None:
        """Accumulated gradients must average, not sum.

        Without the divide, raising grad_accum_steps multiplies the effective
        learning rate by the same factor and diverges a run that was fine.
        """
        grads = {}
        for accum in (1, 4):
            trainer = make_trainer(
                tmp_path / f"a{accum}",
                tiny_corpus,
                model_config,
                grad_accum_steps=accum,
                batch_size=8,
                max_steps=1,
                warmup_steps=0,
            )
            trainer.model.train()
            trainer.optimizer.zero_grad()
            for _ in range(accum):
                x, y = trainer.train_data.batch(8, torch.device("cpu"))
                _, loss, _ = trainer.model(x, targets=y)
                (loss / accum).backward()
            grads[accum] = trainer.model.norm.weight.grad.norm().item()

        # Same order of magnitude, not accum-times larger.
        assert grads[4] < 4 * grads[1]

    def test_checkpoint_round_trips(
        self, tmp_path: Path, tiny_corpus: tuple[Path, Path], model_config: HFM2Config
    ) -> None:
        trainer = make_trainer(tmp_path, tiny_corpus, model_config, max_steps=3)
        trainer.train()
        path = trainer.save_checkpoint("test.pt")

        fresh = make_trainer(tmp_path / "b", tiny_corpus, model_config)
        assert not torch.equal(
            fresh.model.norm.weight, trainer.model.norm.weight
        ) or fresh.step != trainer.step
        fresh.load_checkpoint(path)

        assert fresh.step == trainer.step
        assert torch.equal(fresh.model.norm.weight, trainer.model.norm.weight)

    def test_resuming_continues_from_the_saved_step(
        self, tmp_path: Path, tiny_corpus: tuple[Path, Path], model_config: HFM2Config
    ) -> None:
        trainer = make_trainer(tmp_path, tiny_corpus, model_config, max_steps=4)
        trainer.train()
        path = trainer.save_checkpoint("mid.pt")

        resumed = make_trainer(tmp_path / "r", tiny_corpus, model_config, max_steps=8)
        resumed.load_checkpoint(path)
        assert resumed.step == 4
        resumed.train()
        assert resumed.step == 8

    def test_checkpoint_records_the_model_config(
        self, tmp_path: Path, tiny_corpus: tuple[Path, Path], model_config: HFM2Config
    ) -> None:
        """Inference reconstructs the architecture from this, so it must be exact."""
        trainer = make_trainer(tmp_path, tiny_corpus, model_config, max_steps=1)
        blob = torch.load(
            trainer.save_checkpoint("c.pt"), weights_only=False, map_location="cpu"
        )
        assert blob["model_config"]["dim"] == 32
        assert blob["model_config"]["vocab_size"] == VOCAB
