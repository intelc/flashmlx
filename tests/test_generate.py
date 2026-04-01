# tests/test_generate.py
import mlx.core as mx
from engine.model_config import ModelArgs, Model
from engine.generate import generate


def _make_tiny_model():
    """Create a tiny model for testing (no real weights — random init)."""
    config = {
        "model_type": "llama",
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "vocab_size": 100,
        "head_dim": 8,
        "tie_word_embeddings": True,
    }
    args = ModelArgs.from_dict(config)
    model = Model(args)
    mx.eval(model.parameters())
    return model


class TestGenerate:
    def test_generates_tokens(self):
        model = _make_tiny_model()
        prompt = mx.array([1, 2, 3])
        tokens = list(generate(model, prompt, max_tokens=5))
        assert len(tokens) == 5
        assert all(isinstance(t, int) for t in tokens)

    def test_generates_valid_token_ids(self):
        model = _make_tiny_model()
        prompt = mx.array([1, 2, 3])
        tokens = list(generate(model, prompt, max_tokens=10))
        # All tokens should be in [0, vocab_size)
        assert all(0 <= t < 100 for t in tokens)

    def test_max_tokens_zero(self):
        model = _make_tiny_model()
        prompt = mx.array([1, 2, 3])
        tokens = list(generate(model, prompt, max_tokens=0))
        assert len(tokens) == 0

    def test_temperature_zero_is_deterministic(self):
        model = _make_tiny_model()
        prompt = mx.array([1, 2, 3])
        tokens1 = list(generate(model, prompt, max_tokens=10, temperature=0.0))
        tokens2 = list(generate(model, prompt, max_tokens=10, temperature=0.0))
        assert tokens1 == tokens2

    def test_chunked_prefill(self):
        """Long prompt should work with chunked prefill."""
        model = _make_tiny_model()
        prompt = mx.arange(50)  # 50 tokens
        tokens = list(generate(model, prompt, max_tokens=5, prefill_chunk_size=16))
        assert len(tokens) == 5
