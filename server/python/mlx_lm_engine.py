"""
Pure-Python engine using mlx-lm's model forward pass.

Drop-in replacement for the C++ Engine, exposing the same interface:
    submit_request(request_id, prompt_tokens, max_tokens, temperature)
    poll_tokens() -> list[TokenOutput]
    start() / stop()

Uses mlx-lm's generate_step internally, which produces the optimal MLX graph
for MoE models like Nemotron-30B (80 tok/s vs 37 tok/s from our C++ graph).
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler


@dataclass
class TokenOutput:
    """Mirrors the C++ TokenOutput struct exposed via pybind11."""
    request_id: str
    tokens: list[int]
    done: bool


@dataclass
class _Request:
    """Internal request state."""
    id: str
    prompt_tokens: list[int]
    max_tokens: int
    temperature: float


class MlxLmEngine:
    """
    Python engine backed by mlx-lm's model and generate_step.

    Runs a background thread that processes requests sequentially (N=1),
    using mlx-lm's async_eval pipelining for maximum single-stream throughput.

    For MoE models this matches mlx-lm's native 80 tok/s because we use
    the exact same MLX graph that mlx-lm builds.
    """

    def __init__(self, model_path: str, max_batch_size: int = 8, max_context_len: int = 2048):
        self._model_path = model_path
        self._max_batch_size = max_batch_size
        self._max_context_len = max_context_len

        print(f"[MlxLmEngine] Loading model from {model_path} ...")
        t0 = time.perf_counter()
        self._model, self._tokenizer = load(model_path)
        dt = time.perf_counter() - t0
        print(f"[MlxLmEngine] Model loaded in {dt:.1f}s")

        # EOS token ids for stopping
        if hasattr(self._tokenizer, 'eos_token_id') and self._tokenizer.eos_token_id is not None:
            if isinstance(self._tokenizer.eos_token_id, list):
                self._eos_tokens = set(self._tokenizer.eos_token_id)
            else:
                self._eos_tokens = {self._tokenizer.eos_token_id}
        else:
            self._eos_tokens = set()

        # Request queue and output queue (thread-safe)
        self._request_queue: deque[_Request] = deque()
        self._request_lock = threading.Lock()

        self._output_queue: deque[TokenOutput] = deque()
        self._output_lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._total_requests = 0

    def submit_request(self, request_id: str, prompt_tokens: list[int],
                       max_tokens: int, temperature: float):
        """Submit a generation request. Thread-safe."""
        req = _Request(
            id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        with self._request_lock:
            self._request_queue.append(req)
            self._total_requests += 1

    def poll_tokens(self) -> list[TokenOutput]:
        """Drain all pending token outputs. Thread-safe."""
        results = []
        with self._output_lock:
            while self._output_queue:
                results.append(self._output_queue.popleft())
        return results

    def start(self):
        """Start the background generation loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[MlxLmEngine] Background loop started")

    def stop(self):
        """Stop the background generation loop."""
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10)
        print("[MlxLmEngine] Background loop stopped")

    def ping(self) -> str:
        return "mlx-lm engine ready"

    def get_stats(self):
        return {
            "active_requests": len(self._request_queue),
            "total_requests": self._total_requests,
        }

    def _loop(self):
        """Background thread: process requests one at a time using generate_step."""
        while self._running:
            # Grab next request
            req = None
            with self._request_lock:
                if self._request_queue:
                    req = self._request_queue.popleft()

            if req is None:
                time.sleep(0.001)
                continue

            self._process_request(req)

    def _process_request(self, req: _Request):
        """Run generation for a single request using mlx-lm's generate_step."""
        prompt = mx.array(req.prompt_tokens)

        # Build sampler based on temperature
        if req.temperature <= 0:
            sampler = None  # generate_step defaults to argmax
        else:
            sampler = make_sampler(temp=req.temperature)

        # Create fresh prompt cache for this request
        prompt_cache = make_prompt_cache(self._model)

        # Use generate_step -- this builds the optimal MLX graph with
        # async_eval pipelining (the key to 80 tok/s on MoE models)
        token_buffer = []
        flush_interval = 1  # emit tokens one at a time for low latency

        try:
            for token_id, logprobs in generate_step(
                prompt,
                self._model,
                max_tokens=req.max_tokens,
                sampler=sampler,
                prompt_cache=prompt_cache,
            ):
                # Check EOS
                if token_id in self._eos_tokens:
                    # Flush any buffered tokens, then send done
                    if token_buffer:
                        self._emit(req.id, token_buffer, done=False)
                        token_buffer = []
                    self._emit(req.id, [], done=True)
                    return

                token_buffer.append(token_id)

                if len(token_buffer) >= flush_interval:
                    self._emit(req.id, token_buffer, done=False)
                    token_buffer = []

        except Exception as e:
            print(f"[MlxLmEngine] Error generating for {req.id}: {e}")

        # Flush remaining tokens and mark done
        if token_buffer:
            self._emit(req.id, token_buffer, done=False)
        self._emit(req.id, [], done=True)

    def _emit(self, request_id: str, tokens: list[int], done: bool):
        """Push a TokenOutput to the output queue."""
        out = TokenOutput(request_id=request_id, tokens=tokens, done=done)
        with self._output_lock:
            self._output_queue.append(out)
