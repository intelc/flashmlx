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
