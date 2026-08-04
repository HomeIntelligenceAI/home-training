"""Token stream storage and batching.

The uint16 round-trip test is the one that matters. torch has no uint16 dtype,
so a missing astype turns every id above 32,767 into a different token —
silently, with no error, producing a model that trains on corrupted text.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from home_training.data import TokenDataset, write_tokens


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    path = tmp_path / "train.bin"
    write_tokens(path, list(range(1000)))
    return path


class TestWriteTokens:
    def test_reports_token_count_and_size(self, tmp_path: Path) -> None:
        stats = write_tokens(tmp_path / "t.bin", [1, 2, 3, 4])
        assert stats.tokens == 4
        assert stats.bytes_on_disk == 8  # uint16

    def test_rejects_ids_beyond_uint16(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="uint16"):
            write_tokens(tmp_path / "t.bin", [1, 70000])

    def test_rejects_negative_ids(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="uint16"):
            write_tokens(tmp_path / "t.bin", [1, -1])

    def test_high_ids_survive_the_round_trip(self, tmp_path: Path) -> None:
        """32,768..65,535 must come back unchanged, not wrapped or reinterpreted."""
        path = tmp_path / "t.bin"
        original = [0, 32767, 32768, 65535, 40000]
        write_tokens(path, original)
        data = TokenDataset(path, seq_len=2)
        x, _ = data._gather(torch.tensor([0]), torch.device("cpu"))
        assert x[0].tolist() == original[:2]
        assert np.array(data.tokens).tolist() == original


class TestTokenDataset:
    def test_missing_file_is_reported_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            TokenDataset(tmp_path / "absent.bin", seq_len=8)

    def test_corpus_shorter_than_one_window_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "tiny.bin"
        write_tokens(path, [1, 2, 3])
        with pytest.raises(ValueError, match="at least"):
            TokenDataset(path, seq_len=64)

    def test_batch_has_the_requested_shape(self, corpus: Path) -> None:
        data = TokenDataset(corpus, seq_len=16)
        x, y = data.batch(4, torch.device("cpu"))
        assert x.shape == y.shape == (4, 16)
        assert x.dtype == torch.int64

    def test_targets_are_the_inputs_shifted_by_one(self, corpus: Path) -> None:
        """Next-token prediction. Off-by-one here trains the model to copy."""
        data = TokenDataset(corpus, seq_len=16)
        x, y = data.batch(8, torch.device("cpu"))
        assert torch.equal(x[:, 1:], y[:, :-1])

    def test_sampling_is_reproducible_from_a_seed(self, corpus: Path) -> None:
        data = TokenDataset(corpus, seq_len=16)
        a, _ = data.batch(4, torch.device("cpu"), torch.Generator().manual_seed(7))
        b, _ = data.batch(4, torch.device("cpu"), torch.Generator().manual_seed(7))
        assert torch.equal(a, b)

    def test_sequential_batches_are_deterministic(self, corpus: Path) -> None:
        """Eval loss must be comparable across checkpoints, so order is fixed."""
        data = TokenDataset(corpus, seq_len=16)
        first = data.sequential_batches(2, 3, torch.device("cpu"))
        second = data.sequential_batches(2, 3, torch.device("cpu"))
        assert len(first) == 3
        assert all(torch.equal(a[0], b[0]) for a, b in zip(first, second, strict=True))

    def test_sequential_batches_do_not_overlap(self, corpus: Path) -> None:
        data = TokenDataset(corpus, seq_len=8)
        batches = data.sequential_batches(1, 3, torch.device("cpu"))
        starts = [int(x[0, 0].item()) for x, _ in batches]
        assert starts == [0, 8, 16]


class TestVocabGuard:
    """A stream tokenised with the wrong vocabulary must fail before step 1.

    This is not hypothetical. A stale config pointed a 32,768-vocab model at a
    651K-token stream built with vocab 8,192. It trained happily at 93% GPU
    utilisation, with 75% of the embedding receiving no gradient at all, and
    would have burned seven days producing nothing. Nothing raised.
    """

    def test_ids_beyond_vocab_are_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "big.bin"
        write_tokens(path, list(range(2000)) * 10)
        data = TokenDataset(path, seq_len=16)
        with pytest.raises(ValueError, match="different, larger vocabulary"):
            data.check_against_vocab(vocab_size=1000)

    def test_low_utilisation_is_rejected(self, tmp_path: Path) -> None:
        """The mismatch that produced no error: stream vocab far below model."""
        path = tmp_path / "small.bin"
        write_tokens(path, list(range(200)) * 100)
        data = TokenDataset(path, seq_len=16)
        with pytest.raises(ValueError, match="never be trained"):
            data.check_against_vocab(vocab_size=32768)

    def test_matching_vocab_passes_and_reports(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.bin"
        write_tokens(path, list(range(900)) * 20)
        data = TokenDataset(path, seq_len=16)
        summary = data.check_against_vocab(vocab_size=1000)
        assert "90.0% of vocab" in summary

    def test_the_boundary_id_is_allowed(self, tmp_path: Path) -> None:
        """vocab_size N means valid ids are 0..N-1; N-1 must not be rejected."""
        path = tmp_path / "edge.bin"
        write_tokens(path, list(range(1000)) * 10)
        data = TokenDataset(path, seq_len=16)
        assert data.check_against_vocab(vocab_size=1000)
