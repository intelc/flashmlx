# tests/test_validate.py
import mlx.core as mx
from harness.validate import compute_perplexity, check_correctness


class TestPerplexity:
    def test_perfect_prediction_low_perplexity(self):
        """If the model always predicts the correct next token with high confidence, perplexity is low."""
        vocab_size = 10
        seq_len = 5
        logits = mx.ones((1, seq_len, vocab_size)) * -10.0
        targets = mx.array([[0, 1, 2, 3, 4]])
        # Set correct token logit very high
        for i in range(seq_len):
            logits = logits.at[0, i, targets[0, i].item()].add(20.0)
        ppl = compute_perplexity(logits, targets)
        assert ppl < 2.0  # near-perfect prediction

    def test_uniform_prediction_high_perplexity(self):
        """If the model predicts uniformly, perplexity ≈ vocab_size."""
        vocab_size = 100
        seq_len = 50
        logits = mx.zeros((1, seq_len, vocab_size))  # uniform
        targets = mx.zeros((1, seq_len), dtype=mx.int32)
        ppl = compute_perplexity(logits, targets)
        assert abs(ppl - vocab_size) < 5  # close to 100

    def test_check_correctness_passes(self):
        result = check_correctness(current_ppl=8.04, baseline_ppl=8.0, threshold_pct=1.0)
        assert result.passed is True

    def test_check_correctness_fails(self):
        result = check_correctness(current_ppl=9.0, baseline_ppl=8.0, threshold_pct=1.0)
        assert result.passed is False
        assert result.ppl_delta_pct > 1.0


class TestCheckCorrectness:
    def test_exact_match_passes(self):
        result = check_correctness(current_ppl=8.0, baseline_ppl=8.0, threshold_pct=1.0)
        assert result.passed is True
        assert result.ppl_delta_pct == 0.0

    def test_within_threshold_passes(self):
        result = check_correctness(current_ppl=8.04, baseline_ppl=8.0, threshold_pct=1.0)
        assert result.passed is True

    def test_above_threshold_fails(self):
        result = check_correctness(current_ppl=8.16, baseline_ppl=8.0, threshold_pct=1.0)
        assert result.passed is False
