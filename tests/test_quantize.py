# tests/test_quantize.py
from engine.quantize import QuantConfig


class TestQuantConfig:
    def test_default_config(self):
        config = QuantConfig()
        assert config.bits == 4
        assert config.group_size == 64

    def test_from_model_config_with_quantization(self):
        model_config = {
            "quantization": {"bits": 4, "group_size": 64}
        }
        config = QuantConfig.from_model_config(model_config)
        assert config.bits == 4
        assert config.group_size == 64

    def test_from_model_config_without_quantization(self):
        model_config = {"hidden_size": 768}
        config = QuantConfig.from_model_config(model_config)
        assert config is None

    def test_from_model_config_quantization_config_key(self):
        """Some models use 'quantization_config' instead of 'quantization'."""
        model_config = {
            "quantization_config": {"bits": 8, "group_size": 32}
        }
        config = QuantConfig.from_model_config(model_config)
        assert config.bits == 8
        assert config.group_size == 32
