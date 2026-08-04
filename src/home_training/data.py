"""Token streams for training.

Tokenised corpora are stored as flat binary files of uint16 — one contiguous
stream, no per-document framing. At vocab 32,768 that is 2 bytes per token, so
a 3B-token corpus is 6 GB on disk and is read through a memory map rather than
loaded. The OS page cache does the work, and RAM stops being the limit on
corpus size.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

__all__ = ["TokenDataset", "DataStats", "write_tokens"]

# uint16 holds vocab up to 65,536. The reference vocab is 32,768, so this is
# half the size of uint32 for free. write_tokens enforces the ceiling.
TOKEN_DTYPE = np.uint16
MAX_VOCAB = np.iinfo(TOKEN_DTYPE).max + 1


@dataclass(frozen=True)
class DataStats:
    tokens: int
    bytes_on_disk: int

    def __str__(self) -> str:
        return f"{self.tokens:,} tokens ({self.bytes_on_disk / 1e6:.1f} MB)"


def write_tokens(path: Path, tokens: list[int] | np.ndarray) -> DataStats:
    """Write a token stream to a flat binary file."""
    array = np.asarray(tokens, dtype=np.int64)
    if array.size and (array.max() >= MAX_VOCAB or array.min() < 0):
        raise ValueError(
            f"token ids must fit in uint16 (0..{MAX_VOCAB - 1}); "
            f"got range {array.min()}..{array.max()}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    array.astype(TOKEN_DTYPE).tofile(path)
    return DataStats(tokens=int(array.size), bytes_on_disk=path.stat().st_size)


class TokenDataset:
    """A memory-mapped token stream, sampled as fixed-length windows.

    Training draws windows at uniformly random offsets rather than iterating
    epochs. At corpus scale the distinction is immaterial, and it removes the
    shuffle buffer entirely.
    """

    def __init__(self, path: Path, seq_len: int):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"token file not found: {self.path}")
        # mode="r" keeps this read-only and shareable between processes.
        self.tokens = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        self.seq_len = seq_len
        if len(self.tokens) < seq_len + 1:
            raise ValueError(
                f"{self.path} holds {len(self.tokens)} tokens, need at least "
                f"{seq_len + 1} for one window of seq_len={seq_len}"
            )

    def __len__(self) -> int:
        """Number of distinct windows. One spare token for the shifted target."""
        return len(self.tokens) - self.seq_len

    @property
    def stats(self) -> DataStats:
        return DataStats(len(self.tokens), self.path.stat().st_size)

    def check_against_vocab(self, vocab_size: int, sample: int = 5_000_000) -> str:
        """Verify the token stream matches the model's vocabulary.

        Pointing a model at a stream built with a different tokenizer is
        silent: ids simply mean something else, and if the stream's vocabulary
        is *smaller* the surplus embedding rows never receive a gradient. There
        is no error, no crash, just a model that trains and learns nothing
        useful. This is the check that turns that into a loud failure.

        Returns a human-readable summary, or raises if the stream cannot
        possibly have come from this vocabulary.
        """
        window = np.asarray(self.tokens[: min(len(self.tokens), sample)])
        highest = int(window.max()) if window.size else 0
        distinct = int(np.unique(window).size)

        if highest >= vocab_size:
            raise ValueError(
                f"{self.path} contains token id {highest}, but the model "
                f"vocab_size is {vocab_size}. The stream was tokenised with a "
                f"different, larger vocabulary."
            )

        utilisation = distinct / vocab_size
        summary = (
            f"{distinct:,} distinct ids in the first {window.size:,} tokens "
            f"({utilisation:.1%} of vocab {vocab_size:,})"
        )
        if utilisation < 0.5:
            # Not fatal -- a small corpus legitimately uses few ids -- but at
            # this level the usual cause is a vocabulary mismatch.
            raise ValueError(
                f"{summary}. Over half the embedding would never be trained. "
                f"This almost always means {self.path} was tokenised with a "
                f"different vocabulary than the model expects. Re-tokenise, or "
                f"set the correct vocab_size."
            )
        return summary

    def batch(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of (inputs, next-token targets)."""
        offsets = torch.randint(
            len(self), (batch_size,), generator=generator, device="cpu"
        )
        return self._gather(offsets, device)

    def sequential_batches(
        self, batch_size: int, limit: int, device: torch.device
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Deterministic non-overlapping batches, for evaluation.

        Held-out loss has to be comparable across checkpoints, so the eval set
        is walked in a fixed order rather than sampled.
        """
        stride = self.seq_len
        starts = range(0, len(self) - stride, stride)
        batches = []
        chunk: list[int] = []
        for start in starts:
            chunk.append(start)
            if len(chunk) == batch_size:
                batches.append(self._gather(torch.tensor(chunk), device))
                chunk = []
                if len(batches) >= limit:
                    break
        return batches

    def _gather(
        self, offsets: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # astype(int64) before torch.from_numpy: torch has no uint16 dtype, and
        # a silent reinterpret here would corrupt every id above 32,767.
        x = torch.stack(
            [
                torch.from_numpy(
                    self.tokens[i : i + self.seq_len].astype(np.int64)
                )
                for i in offsets.tolist()
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    self.tokens[i + 1 : i + 1 + self.seq_len].astype(np.int64)
                )
                for i in offsets.tolist()
            ]
        )
        if device.type == "cuda":
            # pin_memory + non_blocking overlaps the host copy with compute.
            return (
                x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True),
            )
        return x.to(device), y.to(device)
