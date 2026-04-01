# FlashMLX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular MLX inference engine for Apple Silicon with an autoresearch-style optimization loop that iteratively discovers faster configurations.

**Architecture:** Two components — (1) FlashMLX Engine: a pipeline of composable modules (attention, KV cache, quantization, generation, model config) that autoresearch modifies, and (2) AutoBench: an immutable benchmark harness + research loop that proposes changes, benchmarks them, and keeps improvements via git. The engine loads HuggingFace models in mlx-community format and runs inference using MLX's Metal backend.

**Tech Stack:** Python 3.12, mlx, mlx-nn, huggingface_hub, transformers (tokenizer only), numpy, pytest

---

## File Structure

```
flashmlx/
├── engine/
│   ├── __init__.py              # Exports: load_model, generate, KVCache, QuantConfig
│   ├── kv_cache.py              # KVCache class with update/get/prefix_match
│   ├── attention.py             # attend() — standard and GQA attention
│   ├── quantize.py              # QuantConfig, quantize_weights()
│   ├── model_config.py          # ModelArgs, Attention, MLP, TransformerBlock, Model, load_model()
│   └── generate.py              # generate() iterator, chunked prefill, sampling
├── harness/
│   ├── __init__.py
│   ├── models.yaml              # Model registry with tiers
│   ├── prepare.py               # One-time setup: download models, compute baselines
│   ├── benchmark.py             # Benchmark protocol: tok/s, TTFT, memory
│   └── validate.py              # Perplexity correctness gate
├── autobench/
│   ├── __init__.py
│   ├── program.md               # Human-authored research directives
│   ├── loop.py                  # Core autoresearch loop
│   └── analyze.py               # Post-hoc Pareto analysis and plots
├── baselines/
│   ├── __init__.py
│   └── compare.py               # Compare against mlx-lm
├── tests/
│   ├── __init__.py
│   ├── test_kv_cache.py
│   ├── test_attention.py
│   ├── test_quantize.py
│   ├── test_model_config.py
│   ├── test_generate.py
│   ├── test_benchmark.py
│   ├── test_validate.py
│   └── test_loop.py
├── pyproject.toml
├── .gitignore
└── docs/superpowers/
    ├── specs/2026-04-01-flashmlx-design.md
    └── plans/2026-04-01-flashmlx-plan.md
```

---

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `engine/__init__.py`
- Create: `harness/__init__.py`
- Create: `autobench/__init__.py`
- Create: `baselines/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "flashmlx"
version = "0.1.0"
description = "Auto-research driven MLX inference engine for Apple Silicon"
requires-python = ">=3.11"
dependencies = [
    "mlx>=0.22.0",
    "huggingface_hub>=0.20.0",
    "transformers>=4.40.0",
    "numpy>=1.26.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-timeout>=2.2",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.pytest.ini_options]
testpaths = ["tests"]
timeout = 120
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
*.egg-info/
.venv/
dist/
build/
harness/baselines.json
autobench/results.tsv
*.log
.DS_Store
models/
```

- [ ] **Step 3: Create empty __init__.py files**

Create empty `engine/__init__.py`, `harness/__init__.py`, `autobench/__init__.py`, `baselines/__init__.py`, `tests/__init__.py`.

- [ ] **Step 4: Create venv and install dependencies**

Run:
```bash
cd /Users/yihengchen/codestuff/aiexperiments/flashmlx
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Expected: All packages install successfully, `python -c "import mlx; print(mlx.__version__)"` prints a version.

- [ ] **Step 5: Verify pytest runs**

Run: `source .venv/bin/activate && python -m pytest tests/ -v`
Expected: "no tests ran" (0 collected), exit code 5 (no tests found — that's fine)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore engine/__init__.py harness/__init__.py autobench/__init__.py baselines/__init__.py tests/__init__.py
git commit -m "feat: project setup with dependencies and structure"
```

---

### Task 2: KV Cache

**Files:**
- Create: `engine/kv_cache.py`
- Create: `tests/test_kv_cache.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_kv_cache.py
import mlx.core as mx
from engine.kv_cache import KVCache


class TestKVCache:
    def test_empty_cache_offset_is_zero(self):
        cache = KVCache()
        assert cache.offset == 0

    def test_update_and_fetch_single_layer(self):
        cache = KVCache()
        # Simulate: batch=1, n_kv_heads=2, seq_len=3, head_dim=4
        keys = mx.ones((1, 2, 3, 4))
        values = mx.ones((1, 2, 3, 4)) * 2
        k_out, v_out = cache.update_and_fetch(keys, values)
        assert k_out.shape == (1, 2, 3, 4)
        assert v_out.shape == (1, 2, 3, 4)
        assert cache.offset == 3

    def test_update_accumulates_tokens(self):
        cache = KVCache()
        k1 = mx.ones((1, 2, 3, 4))
        v1 = mx.ones((1, 2, 3, 4))
        cache.update_and_fetch(k1, v1)

        k2 = mx.ones((1, 2, 1, 4)) * 5
        v2 = mx.ones((1, 2, 1, 4)) * 10
        k_out, v_out = cache.update_and_fetch(k2, v2)
        # Should have 3 + 1 = 4 tokens
        assert k_out.shape == (1, 2, 4, 4)
        assert v_out.shape == (1, 2, 4, 4)
        assert cache.offset == 4
        # Last token should be the new values
        assert mx.allclose(k_out[:, :, -1:, :], mx.ones((1, 2, 1, 4)) * 5).item()
        assert mx.allclose(v_out[:, :, -1:, :], mx.ones((1, 2, 1, 4)) * 10).item()

    def test_multiple_sequential_updates(self):
        cache = KVCache()
        for i in range(10):
            k = mx.ones((1, 4, 1, 8)) * i
            v = mx.ones((1, 4, 1, 8)) * (i + 100)
            k_out, v_out = cache.update_and_fetch(k, v)
        assert cache.offset == 10
        assert k_out.shape == (1, 4, 10, 8)

    def test_pre_allocation_step(self):
        """Cache should pre-allocate in steps to avoid per-token reallocation."""
        cache = KVCache(step=256)
        k = mx.ones((1, 2, 1, 4))
        v = mx.ones((1, 2, 1, 4))
        cache.update_and_fetch(k, v)
        # Internal buffer should be larger than offset
        assert cache.offset == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_kv_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.kv_cache'`

- [ ] **Step 3: Implement KVCache**

```python
# engine/kv_cache.py
import mlx.core as mx


class KVCache:
    """Pre-allocating KV cache for transformer attention layers.

    Stores keys and values as [B, n_kv_heads, seq_len, head_dim] tensors.
    Pre-allocates in chunks of `step` tokens to avoid per-token reallocation.
    """

    def __init__(self, step: int = 256):
        self.step = step
        self.offset = 0
        self._keys = None
        self._values = None
        self._capacity = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Append new keys/values and return the full cached history.

        Args:
            keys: [B, n_kv_heads, new_tokens, head_dim]
            values: [B, n_kv_heads, new_tokens, head_dim]

        Returns:
            Tuple of (all_keys, all_values) each [B, n_kv_heads, total_tokens, head_dim]
        """
        B, H, T, D = keys.shape
        new_offset = self.offset + T

        if self._keys is None:
            # First call — allocate buffer
            self._capacity = max(self.step, new_offset)
            self._keys = mx.zeros((B, H, self._capacity, D))
            self._values = mx.zeros((B, H, self._capacity, D))

        if new_offset > self._capacity:
            # Grow buffer
            new_capacity = self._capacity
            while new_capacity < new_offset:
                new_capacity += self.step
            k_new = mx.zeros((B, H, new_capacity, D))
            v_new = mx.zeros((B, H, new_capacity, D))
            k_new[:, :, : self.offset, :] = self._keys[:, :, : self.offset, :]
            v_new[:, :, : self.offset, :] = self._values[:, :, : self.offset, :]
            self._keys = k_new
            self._values = v_new
            self._capacity = new_capacity

        # Write new tokens into pre-allocated buffer
        self._keys[:, :, self.offset : new_offset, :] = keys
        self._values[:, :, self.offset : new_offset, :] = values
        self.offset = new_offset

        return self._keys[:, :, : self.offset, :], self._values[:, :, : self.offset, :]

    @property
    def state(self):
        """Return current cache state for mx.eval materialization."""
        if self._keys is None:
            return []
        return [self._keys, self._values]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_kv_cache.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add engine/kv_cache.py tests/test_kv_cache.py
git commit -m "feat: KV cache with pre-allocated buffer"
```

---

### Task 3: Attention

**Files:**
- Create: `engine/attention.py`
- Create: `tests/test_attention.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_attention.py
import mlx.core as mx
import mlx.nn as nn
from engine.attention import scaled_dot_product_attention, create_causal_mask
from engine.kv_cache import KVCache


class TestCausalMask:
    def test_causal_mask_shape(self):
        mask = create_causal_mask(seq_len=4, offset=0)
        assert mask.shape == (4, 4)

    def test_causal_mask_is_lower_triangular(self):
        mask = create_causal_mask(seq_len=3, offset=0)
        # Upper triangle should be -inf (masked), lower triangle + diagonal should be 0
        assert mask[0, 1].item() == float("-inf")
        assert mask[0, 2].item() == float("-inf")
        assert mask[1, 2].item() == float("-inf")
        assert mask[0, 0].item() == 0.0
        assert mask[1, 1].item() == 0.0
        assert mask[2, 2].item() == 0.0

    def test_causal_mask_with_offset(self):
        # With offset=5, generating 1 new token that can attend to 5 cached + 1 new = 6
        mask = create_causal_mask(seq_len=1, offset=5)
        assert mask.shape == (1, 6)
        # Single new token can attend to everything
        assert mx.all(mask == 0.0).item()


class TestScaledDotProductAttention:
    def test_output_shape_standard(self):
        B, n_heads, L, D = 1, 4, 8, 16
        q = mx.random.normal((B, n_heads, L, D))
        k = mx.random.normal((B, n_heads, L, D))
        v = mx.random.normal((B, n_heads, L, D))
        mask = create_causal_mask(L, offset=0)
        out = scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask=mask)
        assert out.shape == (B, n_heads, L, D)

    def test_output_shape_gqa(self):
        """GQA: 8 query heads, 2 KV heads — should broadcast correctly."""
        B, L, D = 1, 4, 16
        n_q_heads, n_kv_heads = 8, 2
        q = mx.random.normal((B, n_q_heads, L, D))
        k = mx.random.normal((B, n_kv_heads, L, D))
        v = mx.random.normal((B, n_kv_heads, L, D))
        mask = create_causal_mask(L, offset=0)
        out = scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask=mask)
        assert out.shape == (B, n_q_heads, L, D)

    def test_attention_respects_mask(self):
        """First token should only attend to itself (causal mask)."""
        B, n_heads, L, D = 1, 1, 4, 8
        q = mx.random.normal((B, n_heads, L, D))
        k = mx.random.normal((B, n_heads, L, D))
        # Set value of token 0 to all 1s, rest to all 0s
        v = mx.zeros((B, n_heads, L, D))
        v[:, :, 0:1, :] = mx.ones((B, n_heads, 1, D))
        mask = create_causal_mask(L, offset=0)
        out = scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask=mask)
        # First token output should be exactly [1,1,...,1] since it only attends to token 0
        assert mx.allclose(out[:, :, 0, :], mx.ones((B, n_heads, D)), atol=1e-5).item()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_attention.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.attention'`

- [ ] **Step 3: Implement attention module**

```python
# engine/attention.py
import mlx.core as mx


def create_causal_mask(seq_len: int, offset: int = 0) -> mx.array:
    """Create additive causal attention mask.

    Returns a [seq_len, offset + seq_len] mask where masked positions are -inf
    and allowed positions are 0.
    """
    total_len = offset + seq_len
    # Row indices: the new tokens at positions [offset, offset+seq_len)
    row_idx = mx.arange(offset, offset + seq_len).reshape(-1, 1)
    # Col indices: all positions [0, total_len)
    col_idx = mx.arange(total_len).reshape(1, -1)
    mask = mx.where(col_idx <= row_idx, 0.0, float("-inf"))
    return mask


def scaled_dot_product_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    mask: mx.array | None = None,
) -> mx.array:
    """Scaled dot-product attention with GQA support.

    Uses mx.fast.scaled_dot_product_attention which natively handles
    n_q_heads != n_kv_heads by broadcasting KV heads.

    Args:
        q: [B, n_q_heads, L_q, head_dim]
        k: [B, n_kv_heads, L_kv, head_dim]
        v: [B, n_kv_heads, L_kv, head_dim]
        scale: typically head_dim ** -0.5
        mask: [L_q, L_kv] additive mask (0 = attend, -inf = mask)

    Returns:
        [B, n_q_heads, L_q, head_dim]
    """
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_attention.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add engine/attention.py tests/test_attention.py
git commit -m "feat: attention with causal mask and GQA support"
```

---

### Task 4: Quantization Config

**Files:**
- Create: `engine/quantize.py`
- Create: `tests/test_quantize.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_quantize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.quantize'`

- [ ] **Step 3: Implement quantize module**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_quantize.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add engine/quantize.py tests/test_quantize.py
git commit -m "feat: quantization config with HF config.json parsing"
```

---

### Task 5: Model Config — ModelArgs and Architecture Detection

**Files:**
- Create: `engine/model_config.py`
- Create: `tests/test_model_config.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_model_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.model_config'`

- [ ] **Step 3: Implement ModelArgs and detect_architecture**

```python
# engine/model_config.py
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from engine.attention import scaled_dot_product_attention, create_causal_mask
from engine.kv_cache import KVCache
from engine.quantize import QuantConfig


# Models that use the Llama architecture (GQA + SwiGLU + RMSNorm)
LLAMA_FAMILY = {"llama", "mistral", "qwen2", "qwen2_moe", "gemma", "gemma2", "phi3", "cohere"}


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
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask: mx.array | None = None, cache: KVCache | None = None) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        self.args = args
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
            offset = cache[0].offset if cache else 0
            mask = create_causal_mask(h.shape[1], offset)
        for i, layer in enumerate(self.layers):
            h = layer(h, mask, cache[i] if cache else None)
        h = self.norm(h)
        if self.args.tie_word_embeddings:
            return self.embed_tokens.as_linear(h)
        return self.lm_head(h)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    def sanitize(self, weights: dict) -> dict:
        return {
            k: v for k, v in weights.items()
            if "self_attn.rotary_emb.inv_freq" not in k
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_model_config.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add engine/model_config.py tests/test_model_config.py
git commit -m "feat: model config with Llama architecture and GQA attention"
```

---

### Task 6: Model Loading — load_model()

**Files:**
- Modify: `engine/model_config.py` (add `load_model` function)
- Modify: `tests/test_model_config.py` (add loading tests)

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_model_config.py
import json
import tempfile
import os
import mlx.core as mx
from engine.model_config import load_model


class TestLoadModel:
    def test_load_model_from_dir(self, tmp_path):
        """Test loading from a directory with config.json and safetensors."""
        # Create a minimal config
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
        from engine.model_config import ModelArgs, Model
        args = ModelArgs.from_dict(config)
        model = Model(args)
        mx.eval(model.parameters())

        # Save weights as safetensors
        weights = dict(model.parameters())
        flat = {}
        _flatten_dict(weights, "", flat)
        mx.save_safetensors(str(tmp_path / "model.safetensors"), flat)

        # Load model from directory
        loaded_model, tokenizer = load_model(str(tmp_path))
        assert loaded_model is not None
        # Should be able to forward pass
        input_ids = mx.array([[1, 2, 3]])
        logits = loaded_model(input_ids)
        assert logits.shape == (1, 3, 100)


def _flatten_dict(d, prefix, out):
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_dict(v, key, out)
        else:
            out[key] = v
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_model_config.py::TestLoadModel -v`
Expected: FAIL — `cannot import name 'load_model' from 'engine.model_config'`

- [ ] **Step 3: Add load_model to model_config.py**

Append to `engine/model_config.py`:

```python
import json
import glob as glob_module
from pathlib import Path
from huggingface_hub import snapshot_download


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
        nn.quantize(model, group_size=quant.group_size, bits=quant.bits)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_model_config.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add engine/model_config.py tests/test_model_config.py
git commit -m "feat: model loading from local dir or HuggingFace"
```

---

### Task 7: Generation Loop

**Files:**
- Create: `engine/generate.py`
- Create: `tests/test_generate.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_generate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.generate'`

- [ ] **Step 3: Implement generate**

```python
# engine/generate.py
from typing import Iterator

import mlx.core as mx

from engine.kv_cache import KVCache


def generate(
    model,
    prompt: mx.array,
    max_tokens: int = 256,
    temperature: float = 0.0,
    prefill_chunk_size: int = 2048,
) -> Iterator[int]:
    """Generate tokens autoregressively.

    Args:
        model: Model with __call__(input_ids, cache) -> logits
        prompt: 1D array of token IDs
        max_tokens: Number of tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        prefill_chunk_size: Process prompt in chunks of this size

    Yields:
        Token IDs one at a time
    """
    if max_tokens == 0:
        return

    cache = model.make_cache()

    # Phase 1: Chunked prefill
    prompt = prompt.reshape(1, -1)  # [1, seq_len]
    prompt_len = prompt.shape[1]
    processed = 0

    while prompt_len - processed > 1:
        chunk_size = min(prefill_chunk_size, prompt_len - processed - 1)
        chunk = prompt[:, processed : processed + chunk_size]
        model(chunk, cache=cache)
        mx.eval([c.state for c in cache])
        processed += chunk_size

    # Process last token(s) of prompt to get first logits
    logits = model(prompt[:, processed:], cache=cache)
    logits = logits[:, -1, :]  # [1, vocab_size]

    y = _sample(logits, temperature)
    mx.eval(y)

    for _ in range(max_tokens):
        yield y.item()

        # Compute next token
        logits = model(y.reshape(1, 1), cache=cache)
        logits = logits[:, -1, :]
        y = _sample(logits, temperature)
        mx.eval(y)


def _sample(logits: mx.array, temperature: float) -> mx.array:
    """Sample a token from logits.

    Args:
        logits: [1, vocab_size]
        temperature: 0.0 for greedy, > 0 for sampling

    Returns:
        [1] array with sampled token ID
    """
    if temperature <= 0.0:
        return mx.argmax(logits, axis=-1)
    return mx.random.categorical(logits * (1.0 / temperature))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_generate.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add engine/generate.py tests/test_generate.py
git commit -m "feat: generation loop with chunked prefill and sampling"
```

---

### Task 8: Engine __init__.py Exports and Integration Test

**Files:**
- Modify: `engine/__init__.py`
- Create: `tests/test_integration.py`

- [ ] **Step 1: Update engine/__init__.py with exports**

```python
# engine/__init__.py
from engine.kv_cache import KVCache
from engine.attention import scaled_dot_product_attention, create_causal_mask
from engine.quantize import QuantConfig
from engine.model_config import ModelArgs, Model, load_model, detect_architecture
from engine.generate import generate

__all__ = [
    "KVCache",
    "scaled_dot_product_attention",
    "create_causal_mask",
    "QuantConfig",
    "ModelArgs",
    "Model",
    "load_model",
    "detect_architecture",
    "generate",
]
```

- [ ] **Step 2: Write integration test**

```python
# tests/test_integration.py
import json
import mlx.core as mx
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

    weights = {}
    _flatten(dict(model.parameters()), "", weights)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    return str(tmp_path)


def _flatten(d, prefix, out):
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten(v, key, out)
        else:
            out[key] = v


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
```

- [ ] **Step 3: Run all tests**

Run: `source .venv/bin/activate && python -m pytest tests/ -v`
Expected: All tests PASS (kv_cache, attention, quantize, model_config, generate, integration)

- [ ] **Step 4: Commit**

```bash
git add engine/__init__.py tests/test_integration.py
git commit -m "feat: engine exports and end-to-end integration test"
```

---

### Task 9: Benchmark Harness — models.yaml and benchmark.py

**Files:**
- Create: `harness/models.yaml`
- Create: `harness/benchmark.py`
- Create: `tests/test_benchmark.py`

- [ ] **Step 1: Create models.yaml**

```yaml
# harness/models.yaml
small_tier:
  - name: qwen3-0.6b
    repo: mlx-community/Qwen3-0.6B-4bit
    arch: llama
    params: 0.6B
    role: primary_iteration

  - name: smollm2-1.7b
    repo: mlx-community/SmolLM2-1.7B-Instruct-4bit
    arch: llama
    params: 1.7B
    role: secondary_iteration

medium_tier:
  - name: llama3-8b
    repo: mlx-community/Meta-Llama-3-8B-4bit
    arch: llama
    params: 8B
    role: validation

  - name: mistral-7b
    repo: mlx-community/Mistral-7B-Instruct-v0.3-4bit
    arch: llama
    params: 7B
    role: validation
```

- [ ] **Step 2: Write the failing benchmark tests**

```python
# tests/test_benchmark.py
import json
import mlx.core as mx
from engine import ModelArgs, Model, generate
from harness.benchmark import BenchmarkResult, run_benchmark


def _create_test_model(tmp_path):
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
    weights = {}
    _flatten(dict(model.parameters()), "", weights)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    return str(tmp_path)


def _flatten(d, prefix, out):
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten(v, key, out)
        else:
            out[key] = v


class TestBenchmark:
    def test_result_fields(self, tmp_path):
        model_dir = _create_test_model(tmp_path)
        result = run_benchmark(
            model_path=model_dir,
            prompt_tokens=list(range(1, 17)),  # 16 tokens
            gen_tokens=8,
            num_runs=1,
            warmup_tokens=4,
        )
        assert isinstance(result, BenchmarkResult)
        assert result.tok_s > 0
        assert result.ttft_ms >= 0
        assert result.memory_mb >= 0

    def test_multiple_runs_uses_median(self, tmp_path):
        model_dir = _create_test_model(tmp_path)
        result = run_benchmark(
            model_path=model_dir,
            prompt_tokens=list(range(1, 9)),
            gen_tokens=4,
            num_runs=3,
            warmup_tokens=2,
        )
        assert result.tok_s > 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_benchmark.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.benchmark'`

- [ ] **Step 4: Implement benchmark.py**

```python
# harness/benchmark.py
import time
import statistics
from dataclasses import dataclass

import mlx.core as mx

from engine import load_model, generate


@dataclass
class BenchmarkResult:
    tok_s: float         # Tokens per second (generation)
    ttft_ms: float       # Time to first token in milliseconds
    memory_mb: float     # Peak Metal memory in MB
    tokens_generated: int


def run_benchmark(
    model_path: str,
    prompt_tokens: list[int],
    gen_tokens: int = 256,
    num_runs: int = 3,
    warmup_tokens: int = 20,
    temperature: float = 0.0,
) -> BenchmarkResult:
    """Run the benchmark protocol on a model.

    Args:
        model_path: Path to model directory or HF repo
        prompt_tokens: List of prompt token IDs
        gen_tokens: Number of tokens to generate per run
        num_runs: Number of runs (takes median)
        warmup_tokens: Tokens to generate for warmup (discarded)
        temperature: Sampling temperature

    Returns:
        BenchmarkResult with median metrics across runs
    """
    model, _ = load_model(model_path)
    prompt = mx.array(prompt_tokens)

    # Warmup: prime Metal shader caches
    if warmup_tokens > 0:
        for _ in generate(model, prompt, max_tokens=warmup_tokens, temperature=temperature):
            pass

    tok_s_runs = []
    ttft_runs = []

    for _ in range(num_runs):
        gen_iter = generate(model, prompt, max_tokens=gen_tokens, temperature=temperature)
        tokens_out = []

        # Measure TTFT
        t_start = time.perf_counter()
        first_token = next(gen_iter)
        t_first = time.perf_counter()
        tokens_out.append(first_token)

        # Measure generation throughput
        for token in gen_iter:
            tokens_out.append(token)
        t_end = time.perf_counter()

        ttft_ms = (t_first - t_start) * 1000
        total_time = t_end - t_start
        tok_s = len(tokens_out) / total_time if total_time > 0 else 0

        tok_s_runs.append(tok_s)
        ttft_runs.append(ttft_ms)

    # Peak memory
    memory_mb = mx.metal.get_peak_memory() / (1024 * 1024) if hasattr(mx, "metal") else 0

    return BenchmarkResult(
        tok_s=statistics.median(tok_s_runs),
        ttft_ms=statistics.median(ttft_runs),
        memory_mb=memory_mb,
        tokens_generated=gen_tokens,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_benchmark.py -v`
Expected: All 2 tests PASS

- [ ] **Step 6: Commit**

```bash
git add harness/models.yaml harness/benchmark.py tests/test_benchmark.py
git commit -m "feat: benchmark harness with tok/s, TTFT, memory measurement"
```

---

### Task 10: Validation — Perplexity Gate

**Files:**
- Create: `harness/validate.py`
- Create: `tests/test_validate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_validate.py
import mlx.core as mx
from harness.validate import compute_perplexity, check_correctness


class TestPerplexity:
    def test_perfect_prediction_low_perplexity(self):
        """If the model always predicts the correct next token with high confidence, perplexity is low."""
        # logits where the correct token has very high probability
        vocab_size = 10
        seq_len = 5
        # logits: [1, seq_len, vocab_size]
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
        # 8.04 is 0.5% above 8.0 — within 1% threshold
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
        # 8.04 is 0.5% above 8.0 — within 1% threshold
        result = check_correctness(current_ppl=8.04, baseline_ppl=8.0, threshold_pct=1.0)
        assert result.passed is True

    def test_above_threshold_fails(self):
        # 8.16 is 2% above 8.0 — exceeds 1% threshold
        result = check_correctness(current_ppl=8.16, baseline_ppl=8.0, threshold_pct=1.0)
        assert result.passed is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_validate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'harness.validate'`

- [ ] **Step 3: Implement validate.py**

```python
# harness/validate.py
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class ValidationResult:
    perplexity: float
    ppl_delta_pct: float
    passed: bool


def compute_perplexity(logits: mx.array, targets: mx.array) -> float:
    """Compute perplexity from model logits and target token IDs.

    Args:
        logits: [B, seq_len, vocab_size] — raw logits from model
        targets: [B, seq_len] — ground truth token IDs

    Returns:
        Perplexity (lower is better)
    """
    B, T, V = logits.shape
    # Log-softmax
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    # Gather log-probs of target tokens
    targets_flat = targets.reshape(-1)
    log_probs_flat = log_probs.reshape(-1, V)
    # Index into log_probs for each target
    target_log_probs = mx.take_along_axis(
        log_probs_flat, targets_flat.reshape(-1, 1), axis=1
    ).squeeze(-1)
    # Mean negative log-likelihood
    nll = -mx.mean(target_log_probs)
    ppl = float(mx.exp(nll).item())
    return ppl


def check_correctness(
    current_ppl: float,
    baseline_ppl: float,
    threshold_pct: float = 1.0,
) -> ValidationResult:
    """Check if current perplexity is within threshold of baseline.

    Args:
        current_ppl: Perplexity from current engine configuration
        baseline_ppl: Perplexity from FP16 baseline
        threshold_pct: Maximum allowed degradation in percent

    Returns:
        ValidationResult with pass/fail and delta
    """
    if baseline_ppl == 0:
        ppl_delta_pct = 0.0 if current_ppl == 0 else float("inf")
    else:
        ppl_delta_pct = (current_ppl - baseline_ppl) / baseline_ppl * 100.0
    passed = ppl_delta_pct < threshold_pct
    return ValidationResult(
        perplexity=current_ppl,
        ppl_delta_pct=ppl_delta_pct,
        passed=passed,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_validate.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add harness/validate.py tests/test_validate.py
git commit -m "feat: perplexity validation gate for correctness checking"
```

---

### Task 11: Prepare Script

**Files:**
- Create: `harness/prepare.py`

- [ ] **Step 1: Implement prepare.py**

```python
# harness/prepare.py
"""One-time setup: download models, compute baselines, cache eval data."""

import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import yaml

from engine import load_model, generate
from harness.benchmark import run_benchmark
from harness.validate import compute_perplexity

HARNESS_DIR = Path(__file__).parent
MODELS_YAML = HARNESS_DIR / "models.yaml"
BASELINES_JSON = HARNESS_DIR / "baselines.json"

# Fixed eval prompt (first 128 tokens repeated — will be replaced by WikiText-2 in production)
EVAL_PROMPT_SHORT = list(range(1, 129))   # 128 tokens
EVAL_PROMPT_LONG = list(range(1, 1025))   # 1024 tokens


def get_hardware_info() -> dict:
    """Detect Apple Silicon hardware via system_profiler."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        chip = result.stdout.strip()
    except Exception:
        chip = "unknown"

    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        ram_bytes = int(result.stdout.strip())
        ram_gb = ram_bytes / (1024 ** 3)
    except Exception:
        ram_gb = 0

    return {"chip": chip, "ram_gb": round(ram_gb, 1)}


def load_models_yaml() -> dict:
    """Load the model registry."""
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)


def prepare(tiers: list[str] | None = None):
    """Download models and compute baselines.

    Args:
        tiers: Which tiers to prepare. Default: ["small_tier"].
    """
    if tiers is None:
        tiers = ["small_tier"]

    registry = load_models_yaml()
    hardware = get_hardware_info()
    baselines = {"hardware": hardware, "models": {}}

    for tier in tiers:
        models = registry.get(tier, [])
        for model_info in models:
            name = model_info["name"]
            repo = model_info["repo"]
            print(f"Preparing {name} from {repo}...")

            try:
                result = run_benchmark(
                    model_path=repo,
                    prompt_tokens=EVAL_PROMPT_SHORT,
                    gen_tokens=64,
                    num_runs=3,
                    warmup_tokens=20,
                )
                baselines["models"][name] = {
                    "repo": repo,
                    "tier": tier,
                    "tok_s": result.tok_s,
                    "ttft_ms": result.ttft_ms,
                    "memory_mb": result.memory_mb,
                }
                print(f"  tok/s: {result.tok_s:.1f}, TTFT: {result.ttft_ms:.1f}ms, Memory: {result.memory_mb:.0f}MB")
            except Exception as e:
                print(f"  FAILED: {e}")
                baselines["models"][name] = {"repo": repo, "tier": tier, "error": str(e)}

    with open(BASELINES_JSON, "w") as f:
        json.dump(baselines, f, indent=2)

    print(f"\nBaselines saved to {BASELINES_JSON}")


if __name__ == "__main__":
    tiers = sys.argv[1:] if len(sys.argv) > 1 else None
    prepare(tiers)
```

- [ ] **Step 2: Verify it runs (dry run — no model download yet)**

Run: `source .venv/bin/activate && python -c "from harness.prepare import get_hardware_info; print(get_hardware_info())"`
Expected: Prints hardware info dict like `{'chip': 'Apple M...', 'ram_gb': ...}`

- [ ] **Step 3: Commit**

```bash
git add harness/prepare.py
git commit -m "feat: prepare script for model download and baseline generation"
```

---

### Task 12: AutoBench — program.md and loop.py

**Files:**
- Create: `autobench/program.md`
- Create: `autobench/loop.py`
- Create: `tests/test_loop.py`

- [ ] **Step 1: Create program.md**

```markdown
# FlashMLX AutoBench Research Program

## Objective
Maximize tokens/second on Apple Silicon for Llama-family models using MLX,
while keeping perplexity within 1% of FP16 baseline.

## Composite Score
score = 0.7 * (tok_s / baseline_tok_s) + 0.3 * (baseline_ttft_ms / ttft_ms)

Higher is better. Baseline values from harness/baselines.json.

## Current Cycle: Round 1 — Attention
Focus on engine/attention.py. Try:
- Explore different ways to compose MLX operations for attention
- Experiment with mx.compile() on the attention function
- Try different memory layouts for Q, K, V tensors
- Leverage mx.fast.scaled_dot_product_attention options

## Constraints
- NEVER modify files in harness/ or autobench/
- NEVER install new packages
- All changes must be in engine/
- If an experiment crashes, read the traceback, fix simple bugs, retry once
- If a fix doesn't work, log as crash and move on
- Prefer simplicity: if two approaches get similar tok/s, keep the simpler one
- The engine must remain compatible: load_model() and generate() interfaces unchanged

## NEVER STOP
Run experiments continuously. Do not pause to ask for permission.
```

- [ ] **Step 2: Write the failing loop tests**

```python
# tests/test_loop.py
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from autobench.loop import (
    parse_results_tsv,
    append_result,
    compute_composite_score,
    ExperimentResult,
)


class TestResultsParsing:
    def test_parse_empty_file(self, tmp_path):
        tsv = tmp_path / "results.tsv"
        tsv.write_text("")
        results = parse_results_tsv(str(tsv))
        assert results == []

    def test_parse_results(self, tmp_path):
        tsv = tmp_path / "results.tsv"
        tsv.write_text(
            "commit\tmodule\ttok_s\tttft_ms\tmemory_mb\tperplexity\tppl_delta_pct\tcomposite_score\tstatus\tdescription\n"
            "abc123\tattention\t142.3\t45.2\t4820\t8.12\t0.2\t0.83\tkeep\tfused kernel\n"
        )
        results = parse_results_tsv(str(tsv))
        assert len(results) == 1
        assert results[0].commit == "abc123"
        assert results[0].tok_s == 142.3
        assert results[0].status == "keep"

    def test_append_result(self, tmp_path):
        tsv = tmp_path / "results.tsv"
        result = ExperimentResult(
            commit="def456", module="kv_cache", tok_s=150.0,
            ttft_ms=40.0, memory_mb=5000, perplexity=8.1,
            ppl_delta_pct=0.1, composite_score=0.9,
            status="keep", description="paged cache",
        )
        append_result(str(tsv), result)
        content = tsv.read_text()
        assert "def456" in content
        assert "paged cache" in content


class TestCompositeScore:
    def test_baseline_scores_1_0(self):
        score = compute_composite_score(
            tok_s=100.0, ttft_ms=50.0,
            baseline_tok_s=100.0, baseline_ttft_ms=50.0,
        )
        assert abs(score - 1.0) < 0.01

    def test_double_speed_scores_higher(self):
        score = compute_composite_score(
            tok_s=200.0, ttft_ms=50.0,
            baseline_tok_s=100.0, baseline_ttft_ms=50.0,
        )
        # 0.7 * 2.0 + 0.3 * 1.0 = 1.7
        assert abs(score - 1.7) < 0.01

    def test_half_ttft_scores_higher(self):
        score = compute_composite_score(
            tok_s=100.0, ttft_ms=25.0,
            baseline_tok_s=100.0, baseline_ttft_ms=50.0,
        )
        # 0.7 * 1.0 + 0.3 * 2.0 = 1.3
        assert abs(score - 1.3) < 0.01
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'autobench.loop'`

- [ ] **Step 4: Implement loop.py**

```python
# autobench/loop.py
"""Core autoresearch loop: propose → commit → benchmark → keep/discard."""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

AUTOBENCH_DIR = Path(__file__).parent
RESULTS_TSV = AUTOBENCH_DIR / "results.tsv"
PROGRAM_MD = AUTOBENCH_DIR / "program.md"
PROJECT_ROOT = AUTOBENCH_DIR.parent

TSV_HEADER = "commit\tmodule\ttok_s\tttft_ms\tmemory_mb\tperplexity\tppl_delta_pct\tcomposite_score\tstatus\tdescription"


@dataclass
class ExperimentResult:
    commit: str
    module: str
    tok_s: float
    ttft_ms: float
    memory_mb: float
    perplexity: float
    ppl_delta_pct: float
    composite_score: float
    status: str  # "keep", "discard", "crash", "rejected"
    description: str


def parse_results_tsv(path: str) -> list[ExperimentResult]:
    """Parse results.tsv into a list of ExperimentResult."""
    results = []
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return results

    lines = p.read_text().strip().split("\n")
    for line in lines[1:]:  # skip header
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 10:
            continue
        results.append(ExperimentResult(
            commit=parts[0],
            module=parts[1],
            tok_s=float(parts[2]),
            ttft_ms=float(parts[3]),
            memory_mb=float(parts[4]),
            perplexity=float(parts[5]),
            ppl_delta_pct=float(parts[6]),
            composite_score=float(parts[7]),
            status=parts[8],
            description=parts[9],
        ))
    return results


def append_result(path: str, result: ExperimentResult):
    """Append an experiment result to the TSV file."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        p.write_text(TSV_HEADER + "\n")

    line = "\t".join([
        result.commit, result.module,
        f"{result.tok_s:.1f}", f"{result.ttft_ms:.1f}",
        f"{result.memory_mb:.0f}", f"{result.perplexity:.4f}",
        f"{result.ppl_delta_pct:.2f}", f"{result.composite_score:.4f}",
        result.status, result.description,
    ])
    with open(p, "a") as f:
        f.write(line + "\n")


def compute_composite_score(
    tok_s: float, ttft_ms: float,
    baseline_tok_s: float, baseline_ttft_ms: float,
) -> float:
    """Compute composite score: 0.7 * normalized_tok_s + 0.3 * normalized_ttft.

    Both components are ratios where > 1.0 means improvement over baseline.
    """
    norm_tok_s = tok_s / baseline_tok_s if baseline_tok_s > 0 else 0
    norm_ttft = baseline_ttft_ms / ttft_ms if ttft_ms > 0 else 0
    return 0.7 * norm_tok_s + 0.3 * norm_ttft


def get_git_head() -> str:
    """Get current HEAD commit hash (short)."""
    result = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    return result.stdout.strip().split()[0] if result.stdout.strip() else "unknown"


def git_commit(message: str, files: list[str]):
    """Stage files and commit."""
    for f in files:
        subprocess.run(["git", "add", f], cwd=str(PROJECT_ROOT))
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(PROJECT_ROOT), capture_output=True,
    )


def git_revert_last():
    """Revert the last commit (discard experiment)."""
    subprocess.run(
        ["git", "reset", "--hard", "HEAD~1"],
        cwd=str(PROJECT_ROOT), capture_output=True,
    )


def load_baselines() -> dict:
    """Load baseline metrics from harness/baselines.json."""
    baselines_path = PROJECT_ROOT / "harness" / "baselines.json"
    with open(baselines_path) as f:
        return json.load(f)


def get_best_score(results: list[ExperimentResult]) -> float:
    """Get the best composite score from kept experiments."""
    kept = [r for r in results if r.status == "keep"]
    if not kept:
        return 1.0  # baseline score
    return max(r.composite_score for r in kept)


# Cycling strategy: which module to target based on experiment number
CYCLE_MODULES = [
    ("attention", 20),
    ("kv_cache", 20),
    ("quantize", 20),
    ("generate", 20),
    ("cross-module", 20),
]


def get_target_module(experiment_num: int) -> str:
    """Determine which module to optimize based on experiment number."""
    cycle_len = sum(count for _, count in CYCLE_MODULES)
    pos = experiment_num % cycle_len
    cumulative = 0
    for module, count in CYCLE_MODULES:
        cumulative += count
        if pos < cumulative:
            return module
    return "cross-module"


def run_loop(
    model_name: str = "qwen3-0.6b",
    max_experiments: int | None = None,
    validation_interval: int = 10,
):
    """Run the autoresearch loop.

    Args:
        model_name: Which model from models.yaml to benchmark against
        max_experiments: Stop after N experiments (None = run forever)
        validation_interval: Run medium-model validation every N experiments
    """
    baselines = load_baselines()
    model_baseline = baselines["models"].get(model_name, {})
    baseline_tok_s = model_baseline.get("tok_s", 100.0)
    baseline_ttft_ms = model_baseline.get("ttft_ms", 50.0)
    model_repo = model_baseline.get("repo", "")

    results = parse_results_tsv(str(RESULTS_TSV))
    best_score = get_best_score(results)
    experiment_num = len(results)

    print(f"Starting autobench loop. Baseline: {baseline_tok_s:.1f} tok/s, {baseline_ttft_ms:.1f}ms TTFT")
    print(f"Best score so far: {best_score:.4f} ({experiment_num} prior experiments)")

    while max_experiments is None or experiment_num < max_experiments:
        target_module = get_target_module(experiment_num)
        print(f"\n--- Experiment {experiment_num + 1}: targeting {target_module} ---")

        # The agent interaction would happen here:
        # 1. Read program.md, current module source, results.tsv tail, git log
        # 2. Send to LLM agent
        # 3. Agent returns a diff
        # 4. Apply diff, commit, benchmark, keep/discard
        #
        # For now, this is a placeholder that will be connected to an LLM API.
        print(f"  [Agent interaction needed — target: engine/{target_module}.py]")
        print(f"  Connect an LLM agent to propose modifications to engine/{target_module}.py")
        break  # Exit until agent is connected

        experiment_num += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FlashMLX AutoBench Loop")
    parser.add_argument("--model", default="qwen3-0.6b", help="Model name from models.yaml")
    parser.add_argument("--max-experiments", type=int, default=None, help="Max experiments (default: unlimited)")
    parser.add_argument("--validation-interval", type=int, default=10, help="Medium-model validation interval")
    args = parser.parse_args()
    run_loop(args.model, args.max_experiments, args.validation_interval)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_loop.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add autobench/program.md autobench/loop.py tests/test_loop.py
git commit -m "feat: autobench loop with results tracking, composite scoring, cycling strategy"
```

---

### Task 13: AutoBench — analyze.py

**Files:**
- Create: `autobench/analyze.py`

- [ ] **Step 1: Implement analyze.py**

```python
# autobench/analyze.py
"""Post-hoc analysis of autobench experiments."""

import json
from collections import defaultdict
from pathlib import Path

from autobench.loop import parse_results_tsv, ExperimentResult, RESULTS_TSV

AUTOBENCH_DIR = Path(__file__).parent
PROJECT_ROOT = AUTOBENCH_DIR.parent


def analyze(results_path: str | None = None) -> dict:
    """Analyze experiment results and return summary statistics.

    Returns dict with:
        - total_experiments: int
        - kept: int
        - discarded: int
        - crashed: int
        - best_tok_s: float
        - best_ttft_ms: float
        - best_composite: float
        - improvement_over_baseline: float (ratio)
        - by_module: dict mapping module -> {kept, discarded, best_score}
        - timeline: list of {experiment_num, composite_score, status}
    """
    path = results_path or str(RESULTS_TSV)
    results = parse_results_tsv(path)

    if not results:
        return {"total_experiments": 0, "message": "No experiments found"}

    kept = [r for r in results if r.status == "keep"]
    discarded = [r for r in results if r.status == "discard"]
    crashed = [r for r in results if r.status == "crash"]

    best_tok_s = max((r.tok_s for r in kept), default=0)
    best_ttft = min((r.ttft_ms for r in kept), default=0) if kept else 0
    best_composite = max((r.composite_score for r in kept), default=0)

    # Per-module breakdown
    by_module = defaultdict(lambda: {"kept": 0, "discarded": 0, "crashed": 0, "best_score": 0})
    for r in results:
        mod = by_module[r.module]
        if r.status == "keep":
            mod["kept"] += 1
            mod["best_score"] = max(mod["best_score"], r.composite_score)
        elif r.status == "discard":
            mod["discarded"] += 1
        elif r.status == "crash":
            mod["crashed"] += 1

    # Timeline
    timeline = [
        {"experiment_num": i, "composite_score": r.composite_score, "status": r.status}
        for i, r in enumerate(results)
    ]

    return {
        "total_experiments": len(results),
        "kept": len(kept),
        "discarded": len(discarded),
        "crashed": len(crashed),
        "best_tok_s": best_tok_s,
        "best_ttft_ms": best_ttft,
        "best_composite_score": best_composite,
        "by_module": dict(by_module),
        "timeline": timeline,
    }


def print_report(results_path: str | None = None):
    """Print a human-readable analysis report."""
    stats = analyze(results_path)

    if stats.get("total_experiments", 0) == 0:
        print("No experiments found.")
        return

    print("=" * 60)
    print("FlashMLX AutoBench Analysis Report")
    print("=" * 60)
    print(f"Total experiments: {stats['total_experiments']}")
    print(f"  Kept:      {stats['kept']}")
    print(f"  Discarded: {stats['discarded']}")
    print(f"  Crashed:   {stats['crashed']}")
    print(f"\nBest results:")
    print(f"  tok/s:           {stats['best_tok_s']:.1f}")
    print(f"  TTFT:            {stats['best_ttft_ms']:.1f}ms")
    print(f"  Composite score: {stats['best_composite_score']:.4f}")

    if stats["by_module"]:
        print(f"\nPer-module breakdown:")
        for module, data in sorted(stats["by_module"].items()):
            print(f"  {module}: {data['kept']} kept, {data['discarded']} discarded, "
                  f"{data['crashed']} crashed, best={data['best_score']:.4f}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    print_report(path)
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `source .venv/bin/activate && python -c "from autobench.analyze import analyze; print(analyze())"`
Expected: `{'total_experiments': 0, 'message': 'No experiments found'}`

- [ ] **Step 3: Commit**

```bash
git add autobench/analyze.py
git commit -m "feat: post-hoc analysis with per-module breakdown and timeline"
```

---

### Task 14: Baseline Comparison

**Files:**
- Create: `baselines/compare.py`

- [ ] **Step 1: Implement compare.py**

```python
# baselines/compare.py
"""Compare FlashMLX engine performance against mlx-lm baseline."""

import time
from dataclasses import dataclass

import mlx.core as mx

from engine import load_model, generate


@dataclass
class ComparisonResult:
    engine: str
    tok_s: float
    ttft_ms: float
    memory_mb: float


def benchmark_flashmlx(
    model_path: str,
    prompt_tokens: list[int],
    gen_tokens: int = 256,
) -> ComparisonResult:
    """Benchmark FlashMLX engine."""
    model, _ = load_model(model_path)
    prompt = mx.array(prompt_tokens)

    # Warmup
    for _ in generate(model, prompt, max_tokens=10, temperature=0.0):
        pass

    # Benchmark
    gen_iter = generate(model, prompt, max_tokens=gen_tokens, temperature=0.0)
    t_start = time.perf_counter()
    first = next(gen_iter)
    t_first = time.perf_counter()
    tokens = [first]
    for t in gen_iter:
        tokens.append(t)
    t_end = time.perf_counter()

    memory_mb = mx.metal.get_peak_memory() / (1024 * 1024) if hasattr(mx, "metal") else 0

    return ComparisonResult(
        engine="flashmlx",
        tok_s=len(tokens) / (t_end - t_start),
        ttft_ms=(t_first - t_start) * 1000,
        memory_mb=memory_mb,
    )


def benchmark_mlx_lm(
    model_path: str,
    prompt_tokens: list[int],
    gen_tokens: int = 256,
) -> ComparisonResult | None:
    """Benchmark mlx-lm for comparison. Returns None if mlx-lm is not installed."""
    try:
        import mlx_lm
    except ImportError:
        return None

    model, tokenizer = mlx_lm.load(model_path)

    # Warmup
    prompt_text = tokenizer.decode(prompt_tokens)
    mlx_lm.generate(model, tokenizer, prompt=prompt_text, max_tokens=10, verbose=False)

    # Benchmark
    t_start = time.perf_counter()
    output = mlx_lm.generate(model, tokenizer, prompt=prompt_text, max_tokens=gen_tokens, verbose=False)
    t_end = time.perf_counter()

    memory_mb = mx.metal.get_peak_memory() / (1024 * 1024) if hasattr(mx, "metal") else 0
    num_tokens = len(tokenizer.encode(output)) - len(prompt_tokens)

    return ComparisonResult(
        engine="mlx-lm",
        tok_s=num_tokens / (t_end - t_start),
        ttft_ms=0,  # mlx-lm doesn't expose TTFT easily in non-streaming mode
        memory_mb=memory_mb,
    )


def compare(model_path: str, prompt_tokens: list[int], gen_tokens: int = 256):
    """Run comparison and print results."""
    print(f"Benchmarking {model_path}...")
    print(f"  Prompt: {len(prompt_tokens)} tokens, Generate: {gen_tokens} tokens\n")

    flash_result = benchmark_flashmlx(model_path, prompt_tokens, gen_tokens)
    print(f"  FlashMLX:  {flash_result.tok_s:.1f} tok/s, TTFT: {flash_result.ttft_ms:.1f}ms, "
          f"Memory: {flash_result.memory_mb:.0f}MB")

    mlx_result = benchmark_mlx_lm(model_path, prompt_tokens, gen_tokens)
    if mlx_result:
        print(f"  mlx-lm:    {mlx_result.tok_s:.1f} tok/s, Memory: {mlx_result.memory_mb:.0f}MB")
        speedup = flash_result.tok_s / mlx_result.tok_s if mlx_result.tok_s > 0 else 0
        print(f"\n  Speedup: {speedup:.2f}x")
    else:
        print("  mlx-lm: not installed (pip install mlx-lm to compare)")


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-0.6B-4bit"
    compare(model, list(range(1, 129)))
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `source .venv/bin/activate && python -c "from baselines.compare import compare; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add baselines/compare.py
git commit -m "feat: baseline comparison against mlx-lm"
```

---

### Task 15: Final Integration — Run All Tests and Verify

**Files:** None (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `source .venv/bin/activate && python -m pytest tests/ -v --tb=short`
Expected: All tests PASS (approximately 25+ tests across all test files)

- [ ] **Step 2: Verify engine imports work end-to-end**

Run:
```bash
source .venv/bin/activate && python -c "
from engine import load_model, generate, KVCache, QuantConfig, ModelArgs, Model
from harness.benchmark import run_benchmark, BenchmarkResult
from harness.validate import compute_perplexity, check_correctness
from autobench.loop import compute_composite_score, parse_results_tsv
from autobench.analyze import analyze
from baselines.compare import benchmark_flashmlx
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 3: Verify the project structure matches the spec**

Run: `find . -name "*.py" -o -name "*.yaml" -o -name "*.md" | grep -v __pycache__ | grep -v .venv | sort`

Expected output should include all files from the file structure above.

- [ ] **Step 4: Commit any fixups**

If any tests needed fixing, commit the fixes:
```bash
git add -u
git commit -m "fix: test suite cleanup and final integration verification"
```

- [ ] **Step 5: Push to remote**

```bash
git push origin main
```
