"""Tests for the classifier architectures in baseline/model.py."""

import torch

from model import CNNClassifier, CNNBPFClassifier, CNNLSTMClassifier


class TestCNNClassifier:
    def test_forward_pass_shape(self):
        """CNN-only forward: (batch, 1, 128, 128) -> (batch, 4)."""
        model = CNNClassifier(n_classes=4)
        x = torch.randn(2, 1, 128, 128)
        logits = model(x)
        assert logits.shape == (2, 4)
        assert torch.all(torch.isfinite(logits))


class TestCNNBPFClassifier:
    def test_forward_pass_shape(self):
        """CNN + BFP forward: spectrogram + 3-D BFP -> class logits."""
        model = CNNBPFClassifier(n_classes=4, bfp_dim=3)
        x = torch.randn(2, 1, 128, 128)
        bfp = torch.randn(2, 3)
        logits = model(x, bfp)
        assert logits.shape == (2, 4)
        assert torch.all(torch.isfinite(logits))


class TestCNNLSTMClassifier:
    def test_forward_pass_shape(self):
        """CNN + LSTM + BFP forward: sequence of frames -> class logits."""
        model = CNNLSTMClassifier(n_classes=4, bfp_dim=3, seq_len=10)
        # (batch=2, seq=10, channel=1, H=128, W=128)
        x = torch.randn(2, 10, 1, 128, 128)
        bfp = torch.randn(2, 10, 3)
        logits = model(x, bfp)
        assert logits.shape == (2, 4)
        assert torch.all(torch.isfinite(logits))

    def test_parameter_count_is_in_expected_range(self):
        """The full CNN+LSTM+BFP model has ~0.48M params (documented size)."""
        model = CNNLSTMClassifier(n_classes=4, bfp_dim=3, seq_len=10)
        total = sum(p.numel() for p in model.parameters())
        assert 300_000 < total < 700_000, (
            f"Unexpected model size: {total:,} parameters. "
            f"Documented range is ~0.48M (baseline/README.md)."
        )

    def test_frame_order_matters_not_mathematically(self):
        """
        Mathematical sanity check: after training, frame order should be
        largely irrelevant (the attribution result). Before training, the
        LSTM's hidden state is random and this test cannot pre-check the
        behavioural claim. We instead check that the model accepts
        permuted frame orders without erroring, which the attribution
        harness depends on.
        """
        torch.manual_seed(0)
        model = CNNLSTMClassifier(n_classes=4, bfp_dim=3, seq_len=10)
        model.eval()
        x = torch.randn(1, 10, 1, 128, 128)
        bfp = torch.randn(1, 10, 3)
        perm = torch.randperm(10)
        with torch.no_grad():
            a = model(x, bfp)
            b = model(x[:, perm], bfp[:, perm])
        # Both produce valid output; we don't assert equality (they differ
        # because the LSTM sees different orderings).
        assert a.shape == b.shape == (1, 4)
