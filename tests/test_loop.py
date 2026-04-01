# tests/test_loop.py
from pathlib import Path
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
