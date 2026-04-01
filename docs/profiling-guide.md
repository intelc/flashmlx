# Profiling Guide for FlashMLX

Tools and techniques for profiling MLX inference on Apple Silicon.

## Available Profiling Tools

### 1. mactop (System-Level GPU Monitoring)

Real-time GPU utilization, power, frequency, and thermal monitoring.

```bash
# Install
brew install mactop

# Interactive TUI
sudo mactop

# Headless mode for scripting (JSON output)
mactop --headless -i 200 > metrics.jsonl

# Parse results
python3 -c "
import json
for line in open('metrics.jsonl'):
    d = json.loads(line)
    m = d['soc_metrics']
    print(f'GPU: {m[\"gpu_active\"]:.0f}% active, {m[\"gpu_power\"]:.1f}W, {m[\"gpu_freq_mhz\"]}MHz')
"
```

Key metrics:
- `gpu_active` — GPU utilization percentage (91% during our inference = good)
- `gpu_power` — GPU power draw (24W on M1 Max during inference, max ~40W)
- `gpu_freq_mhz` — GPU clock frequency
- `soc_temp` — SoC temperature

### 2. mx.metal.start_capture (Metal GPU Trace)

Captures a detailed GPU trace openable in Xcode Metal Debugger. Shows per-kernel timing, GPU occupancy, shader performance, and memory access patterns.

```bash
# IMPORTANT: Must set MTL_CAPTURE_ENABLED=1 before importing mlx
MTL_CAPTURE_ENABLED=1 python3 -c "
import mlx.core as mx
from engine import load_model, generate

model, _ = load_model('mlx-community/Qwen3-0.6B-4bit')
prompt = mx.array(list(range(1, 33)))

# Warmup first (don't capture warmup)
for _ in generate(model, prompt, max_tokens=10, temperature=0.0): pass

# Capture during inference
mx.metal.start_capture('/tmp/inference.gputrace')
tokens = list(generate(model, prompt, max_tokens=16, temperature=0.0))
mx.metal.stop_capture()
"

# Open in Xcode Metal Debugger
open /tmp/inference.gputrace
```

The trace shows:
- Individual Metal compute kernel dispatches
- Per-kernel execution time
- GPU occupancy and ALU utilization
- Memory bandwidth per kernel
- Shader source with cost annotations

**Tip:** Keep captures short (16-32 tokens) — traces are large (~500MB for 16 tokens).

### 3. xctrace (Command-Line Instruments)

Apple's Instruments CLI. Can capture Metal System Trace for system-wide GPU activity.

```bash
# List available templates
xctrace list templates | grep -i metal
# Output: Metal System Trace

# Record a trace (attach to running process)
xctrace record --template "Metal System Trace" \
    --attach <PID> --time-limit 5s \
    --output trace.trace --no-prompt

# Record for all processes
xctrace record --template "Metal System Trace" \
    --all-processes --time-limit 5s \
    --output trace.trace --no-prompt

# Export trace data
xctrace export --input trace.trace --toc  # list available schemas
xctrace export --input trace.trace \
    --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]'
```

Available schemas in Metal System Trace:
- `metal-gpu-intervals` — GPU execution intervals (vertex/fragment/compute)
- `metal-gpu-state-intervals` — GPU active/idle states
- `metal-driver-intervals` — Metal driver processing times
- `metal-application-intervals` — Per-app Metal activity
- `gpu-performance-state-intervals` — GPU P-state changes
- `metal-gpu-counter-intervals` — GPU performance counters (if captured)

**Limitation:** MLX's Metal compute dispatches don't show up in standard xctrace
Metal System Trace because MLX uses its own Metal command buffer management.
Use `mx.metal.start_capture()` instead for MLX-specific profiling.

### 4. MLX Memory APIs

```python
import mlx.core as mx

# Memory usage
print(f'Active: {mx.get_active_memory() / 1e9:.1f} GB')
print(f'Peak:   {mx.get_peak_memory() / 1e9:.1f} GB')
print(f'Cache:  {mx.get_cache_memory() / 1e9:.1f} GB')

# Control memory
mx.set_cache_limit(4 * 1024**3)   # 4GB cache
mx.set_memory_limit(32 * 1024**3)  # 32GB total
mx.clear_cache()
mx.reset_peak_memory()
```

### 5. Python-Level Timing

For quick per-operation profiling without GPU trace overhead:

```python
import time, mlx.core as mx

# Time a single operation
t0 = time.perf_counter()
result = some_operation(x)
mx.eval(result)  # Force evaluation
t1 = time.perf_counter()
print(f'{(t1-t0)*1000:.1f}ms')
```

**Important:** Always call `mx.eval()` before timing to force GPU execution.
MLX is lazy — without eval, you're timing graph construction, not execution.

## Key Insights from Profiling

### Apple Silicon Architecture (from WWDC25)

- **Decode phase is memory-bandwidth bound** — reading model weights dominates
- **Prefill phase is compute bound** — large matmul benefits from parallelism
- **M1 Max: 400 GB/s bandwidth** → theoretical max ~89 tok/s for 8B model
- **Operation fusion reduces kernel dispatch overhead** — use `mx.compile` or `mx.fast.*`
- **Neural accelerators (M5 only)** accelerate matmul 4-8x — not available on M1-M4

### What We Measured

On M1 Max with Qwen3-0.6B-4bit:
- GPU utilization: **91%** during inference (mactop)
- GPU power: **24W** (60% of max 40W TDP)
- GPU frequency: **1294 MHz** (max boost)
- Per-token decode: **8.4ms** in C++ scheduler, **4.0ms** in Python engine
- The 2x gap is from C++ graph construction overhead vs Python's nn.Module caching

### Common Pitfalls

1. **dtype mismatch** — KV cache was float16 but weights were bfloat16, causing implicit conversion kernels (+56% overhead)
2. **mx::eval inside forward pass** — extracting array values mid-graph forces GPU sync (28x per forward for array-offset KV cache)
3. **Pre-allocated cache overhead** — `slice_update` for KV cache is slower than `concat` for small models on MLX
4. **Graph construction cost** — building 280+ MLX ops per forward pass has measurable C++ overhead
