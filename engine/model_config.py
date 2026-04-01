# engine/model_config.py
import json
import glob as glob_module
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from functools import partial

from engine.attention import scaled_dot_product_attention, create_causal_mask
from engine.fused_ops import fused_residual_rms_norm, fused_rms_norm
from engine.kv_cache import KVCache, PreAllocKVCache
from engine.quantize import QuantConfig


@partial(mx.compile, shapeless=True)
def _swiglu(gate, x):
    return nn.silu(gate) * x


# Models that use the Llama architecture (GQA + SwiGLU + RMSNorm)
LLAMA_FAMILY = {"llama", "mistral", "qwen2", "qwen3", "qwen2_moe", "gemma", "gemma2", "phi3", "cohere"}


def detect_architecture(config: dict) -> str:
    """Map HF model_type to our architecture family."""
    model_type = config.get("model_type", "")
    if model_type in LLAMA_FAMILY:
        return "llama"
    raise ValueError(f"Unsupported model_type: {model_type}")


@dataclass
class ModelArgs:
    """Model hyperparameters parsed from HF config.json."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: dict | None = None
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False

    @classmethod
    def from_dict(cls, config: dict) -> "ModelArgs":
        num_attention_heads = config["num_attention_heads"]
        hidden_size = config["hidden_size"]
        num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
        head_dim = config.get("head_dim", hidden_size // num_attention_heads)
        return cls(
            model_type=config.get("model_type", "llama"),
            hidden_size=hidden_size,
            num_hidden_layers=config["num_hidden_layers"],
            intermediate_size=config["intermediate_size"],
            num_attention_heads=num_attention_heads,
            rms_norm_eps=config.get("rms_norm_eps", 1e-5),
            vocab_size=config["vocab_size"],
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_theta=config.get("rope_theta", 10000.0),
            rope_traditional=config.get("rope_traditional", False),
            rope_scaling=config.get("rope_scaling"),
            tie_word_embeddings=config.get("tie_word_embeddings", True),
            attention_bias=config.get("attention_bias", False),
            mlp_bias=config.get("mlp_bias", False),
        )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim ** -0.5

        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=args.attention_bias)
        self.rope = nn.RoPE(
            self.head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: KVCache | None = None) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=args.mlp_bias)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=args.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(_swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._eps = args.rms_norm_eps

    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: KVCache | None = None) -> mx.array:
        # Fused: input_layernorm then attention
        normed = fused_rms_norm(x, self.input_layernorm.weight, self._eps)
        attn_out = self.self_attn(normed, mask, cache)
        # Fused: residual add + post_attention_layernorm
        h, normed2 = fused_residual_rms_norm(x, attn_out, self.post_attention_layernorm.weight, self._eps)
        # MLP
        mlp_out = self.mlp(normed2)
        return h + mlp_out


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        self.args = args
        self._eps = args.rms_norm_eps
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        h = self.embed_tokens(input_ids)
        mask = None
        if h.shape[1] > 1:
            if cache is None or cache[0].offset == 0:
                mask = "causal"
            else:
                offset = cache[0].offset
                mask = create_causal_mask(h.shape[1], offset)
        if cache is None:
            cache = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            h = layer(h, mask, cache[i])
        h = fused_rms_norm(h, self.norm.weight, self._eps)
        if self.args.tie_word_embeddings:
            return self.embed_tokens.as_linear(h)
        return self.lm_head(h)

    def make_cache(self, prealloc: bool = False) -> list:
        cls = PreAllocKVCache if prealloc else KVCache
        return [cls() for _ in self.layers]

    def sanitize(self, weights: dict) -> dict:
        sanitized = {}
        for k, v in weights.items():
            if "self_attn.rotary_emb.inv_freq" in k:
                continue
            # Strip "model." prefix from HF weight keys
            if k.startswith("model."):
                k = k[len("model."):]
            sanitized[k] = v
        return sanitized


def load_model(
    path_or_repo: str,
) -> tuple["Model", any]:
    """Load a model from a local directory or HuggingFace repo.

    Args:
        path_or_repo: Local path or HF repo ID (e.g. "mlx-community/Qwen3-0.6B-4bit")

    Returns:
        Tuple of (model, tokenizer). tokenizer is None if transformers is not available
        or tokenizer files are missing.
    """
    model_path = Path(path_or_repo)
    if not model_path.exists():
        model_path = Path(snapshot_download(
            path_or_repo,
            allow_patterns=["*.json", "model*.safetensors", "tokenizer.model", "*.tiktoken", "*.txt"],
        ))

    # Load config
    with open(model_path / "config.json") as f:
        config = json.load(f)

    # Detect architecture and build model
    detect_architecture(config)  # raises if unsupported
    args = ModelArgs.from_dict(config)
    model = Model(args)

    # Load weights from safetensors
    weight_files = sorted(glob_module.glob(str(model_path / "model*.safetensors")))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    # Sanitize weight keys
    weights = model.sanitize(weights)

    # Apply quantization if model is quantized
    quant = QuantConfig.from_model_config(config)
    if quant is not None:
        # Check if embeddings are quantized in the weights file
        embed_quantized = "embed_tokens.scales" in weights
        nn.quantize(
            model,
            group_size=quant.group_size,
            bits=quant.bits,
            class_predicate=lambda _, m: (
                hasattr(m, "to_quantized")
                and (embed_quantized or not isinstance(m, nn.Embedding))
            ),
        )

    # Load weights into model
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    # Load tokenizer
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    except Exception:
        pass

    return model, tokenizer
