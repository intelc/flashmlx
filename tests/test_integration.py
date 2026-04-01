# tests/test_integration.py
import json
import mlx.core as mx
from mlx.utils import tree_flatten
from engine import load_model, generate, Model, ModelArgs


def _create_test_model_dir(tmp_path):
    """Create a minimal model directory with config and weights."""
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
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)

    args = ModelArgs.from_dict(config)
    model = Model(args)
    mx.eval(model.parameters())

    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    return str(tmp_path)


class TestEndToEnd:
    def test_load_and_generate(self, tmp_path):
        model_dir = _create_test_model_dir(tmp_path)
        model, _ = load_model(model_dir)
        prompt = mx.array([1, 2, 3, 4, 5])
        tokens = list(generate(model, prompt, max_tokens=10, temperature=0.0))
        assert len(tokens) == 10
        assert all(0 <= t < 100 for t in tokens)

    def test_deterministic_generation(self, tmp_path):
        model_dir = _create_test_model_dir(tmp_path)
        model, _ = load_model(model_dir)
        prompt = mx.array([1, 2, 3])
        t1 = list(generate(model, prompt, max_tokens=20, temperature=0.0))
        t2 = list(generate(model, prompt, max_tokens=20, temperature=0.0))
        assert t1 == t2
