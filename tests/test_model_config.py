# tests/test_model_config.py
from engine.model_config import ModelArgs, detect_architecture


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
