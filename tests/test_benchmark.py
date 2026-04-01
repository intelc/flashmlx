# tests/test_benchmark.py
import json
import mlx.core as mx
from mlx.utils import tree_flatten
from engine import ModelArgs, Model
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
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    return str(tmp_path)


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
