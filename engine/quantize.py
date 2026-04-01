# engine/quantize.py
from dataclasses import dataclass


@dataclass
class QuantConfig:
    """Quantization configuration for model weights."""

    bits: int = 4
    group_size: int = 64

    @classmethod
    def from_model_config(cls, config: dict) -> "QuantConfig | None":
        """Extract quantization config from HF config.json dict.

        Returns None if the model is not quantized.
        """
        quant = config.get("quantization") or config.get("quantization_config")
        if quant is None:
            return None
        return cls(
            bits=quant.get("bits", 4),
            group_size=quant.get("group_size", 64),
        )
