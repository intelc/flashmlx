# tests/test_model_config.py
import json
import mlx.core as mx
from mlx.utils import tree_flatten
from engine.model_config import ModelArgs, Model, detect_architecture, load_model


class TestModelArgs:
    def test_from_dict_llama(self):
        config = {
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "intermediate_size": 5632,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-5,
            "vocab_size": 32000,
            "rope_theta": 10000.0,
        }
        args = ModelArgs.from_dict(config)
        assert args.hidden_size == 2048
        assert args.num_attention_heads == 32
        assert args.num_key_value_heads == 8
        assert args.head_dim == 64  # 2048 // 32

    def test_from_dict_defaults(self):
        config = {
            "model_type": "llama",
            "hidden_size": 768,
            "num_hidden_layers": 6,
            "intermediate_size": 2048,
            "num_attention_heads": 12,
            "rms_norm_eps": 1e-5,
            "vocab_size": 32000,
        }
        args = ModelArgs.from_dict(config)
        # num_key_value_heads defaults to num_attention_heads (MHA)
        assert args.num_key_value_heads == 12
        assert args.head_dim == 64

    def test_explicit_head_dim(self):
        config = {
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "intermediate_size": 5632,
            "num_attention_heads": 32,
            "head_dim": 128,
            "rms_norm_eps": 1e-5,
            "vocab_size": 32000,
        }
        args = ModelArgs.from_dict(config)
        assert args.head_dim == 128


class TestDetectArchitecture:
    def test_llama(self):
        assert detect_architecture({"model_type": "llama"}) == "llama"

    def test_qwen2(self):
        assert detect_architecture({"model_type": "qwen2"}) == "llama"

    def test_mistral(self):
        assert detect_architecture({"model_type": "mistral"}) == "llama"

    def test_unknown_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unsupported"):
            detect_architecture({"model_type": "unknown_model_xyz"})


class TestLoadModel:
    def test_load_model_from_dir(self, tmp_path):
        """Test loading from a directory with config.json and safetensors."""
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
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Create model, save its weights, then reload
        args = ModelArgs.from_dict(config)
        model = Model(args)
        mx.eval(model.parameters())

        # Save weights as safetensors using tree_flatten
        flat = dict(tree_flatten(model.parameters()))
        mx.save_safetensors(str(tmp_path / "model.safetensors"), flat)

        # Load model from directory
        loaded_model, tokenizer = load_model(str(tmp_path))
        assert loaded_model is not None
        # Should be able to forward pass
        input_ids = mx.array([[1, 2, 3]])
        logits = loaded_model(input_ids)
        assert logits.shape == (1, 3, 100)
