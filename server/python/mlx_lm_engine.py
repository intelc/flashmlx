"""
Pure-Python engine using mlx-lm's model forward pass.

Two modes:
- Sequential (default): Uses generate_step for single-request, ~60 tok/s
- Batched: Batches multiple active requests into one forward for throughput scaling

Drop-in replacement for the C++ Engine, exposing the same interface:
    submit_request(request_id, prompt_tokens, max_tokens, temperature)
    poll_tokens() -> list[TokenOutput]
    start() / stop()
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List

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


@dataclass
class _ActiveRequest:
    """A request currently being decoded."""
    id: str
    max_tokens: int
    temperature: float
    generated_count: int = 0
    last_token: int = 0
    cache: Optional[list] = None  # mlx-lm prompt cache


class MlxLmEngine:
    """
    Python engine backed by mlx-lm's model and generate_step.

    Supports two modes:
    - batched=False (default): Sequential processing via generate_step.
      Best single-request latency (~60 tok/s per request).
    - batched=True: Batched decode for multi-request throughput scaling.
      Admits new requests continuously, decodes all active requests in
      one forward pass per step.
    """

    def __init__(self, model_path: str, max_batch_size: int = 8,
                 max_context_len: int = 2048, batched: bool = False):
        self._model_path = model_path
        self._max_batch_size = max_batch_size
        self._max_context_len = max_context_len
        self._batched = batched

        print(f"[MlxLmEngine] Loading model from {model_path} ...")
        t0 = time.perf_counter()
        self._model, self._tokenizer = load(model_path)
        dt = time.perf_counter() - t0
        print(f"[MlxLmEngine] Model loaded in {dt:.1f}s (batched={batched})")

        # EOS token ids
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
        req = _Request(id=request_id, prompt_tokens=prompt_tokens,
                       max_tokens=max_tokens, temperature=temperature)
        with self._request_lock:
            self._request_queue.append(req)
            self._total_requests += 1

    def poll_tokens(self) -> list[TokenOutput]:
        results = []
        with self._output_lock:
            while self._output_queue:
                results.append(self._output_queue.popleft())
        return results

    def start(self):
        if self._running:
            return
        self._running = True
        loop = self._batched_loop if self._batched else self._sequential_loop
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        mode = "batched" if self._batched else "sequential"
        print(f"[MlxLmEngine] Background loop started ({mode})")

    def stop(self):
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

    # ---------------------------------------------------------------
    # Sequential mode (original single-request generate_step)
    # ---------------------------------------------------------------

    def _sequential_loop(self):
        while self._running:
            req = None
            with self._request_lock:
                if self._request_queue:
                    req = self._request_queue.popleft()
            if req is None:
                time.sleep(0.001)
                continue
            self._process_sequential(req)

    def _process_sequential(self, req: _Request):
        prompt = mx.array(req.prompt_tokens)
        sampler = None if req.temperature <= 0 else make_sampler(temp=req.temperature)
        prompt_cache = make_prompt_cache(self._model)

        try:
            for token_id, logprobs in generate_step(
                prompt, self._model, max_tokens=req.max_tokens,
                sampler=sampler, prompt_cache=prompt_cache,
            ):
                if token_id in self._eos_tokens:
                    self._emit(req.id, [], done=True)
                    return
                self._emit(req.id, [token_id], done=False)
        except Exception as e:
            print(f"[MlxLmEngine] Error: {e}")
        self._emit(req.id, [], done=True)

    # ---------------------------------------------------------------
    # Batched mode: continuous batching for throughput scaling
    # ---------------------------------------------------------------

    def _batched_loop(self):
        """Continuous batching: prefill new requests, batch-decode active ones."""
        active: list[_ActiveRequest] = []

        while self._running:
            # 1. Admit pending requests — batch-prefill if same prompt length
            new_reqs = []
            while len(active) + len(new_reqs) < self._max_batch_size:
                req = None
                with self._request_lock:
                    if self._request_queue:
                        req = self._request_queue.popleft()
                if req is None:
                    break
                new_reqs.append(req)

            if new_reqs:
                # Check if all new requests have same prompt (common benchmark case)
                same_prompt = all(r.prompt_tokens == new_reqs[0].prompt_tokens for r in new_reqs)
                if same_prompt and len(new_reqs) > 1:
                    # Batched prefill
                    new_active = self._batched_prefill(new_reqs)
                    active.extend(new_active)
                else:
                    # Individual prefill
                    for req in new_reqs:
                        ar = self._prefill_request(req)
                        if ar is not None:
                            active.append(ar)

            if not active:
                time.sleep(0.001)
                continue

            # 2. Batch decode: one forward pass for all active requests
            done_indices = self._batch_decode_step(active)

            # 3. Retire completed requests (reverse order to preserve indices)
            for idx in sorted(done_indices, reverse=True):
                ar = active.pop(idx)
                self._emit(ar.id, [], done=True)

    def _batched_prefill(self, reqs: list[_Request]) -> list[_ActiveRequest]:
        """Batch-prefill multiple requests with the same prompt into one forward pass."""
        B = len(reqs)
        try:
            cache = make_prompt_cache(self._model)
            # Batched prompt: [B, seq_len]
            prompt = mx.array([reqs[0].prompt_tokens] * B)
            logits = self._model(prompt, cache=cache)
            mx.eval([c.state for c in cache])

            # Sample first token per request
            last_logits = logits[:, -1, :]  # [B, vocab]
            results = []
            for b in range(B):
                req = reqs[b]
                rl = last_logits[b:b+1]
                if req.temperature <= 0:
                    token = mx.argmax(rl, axis=-1)
                else:
                    token = mx.random.categorical(rl / req.temperature)
                mx.eval(token)
                tok_id = token.item()

                if tok_id in self._eos_tokens:
                    self._emit(req.id, [], done=True)
                    continue

                self._emit(req.id, [tok_id], done=False)
                ar = _ActiveRequest(
                    id=req.id, max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    generated_count=1, last_token=tok_id, cache=cache,
                )
                ar._batch_cache_ref = cache  # shared batch cache
                results.append(ar)
            return results
        except Exception as e:
            print(f"[MlxLmEngine] Batched prefill error: {e}")
            # Fall back to individual prefill
            results = []
            for req in reqs:
                ar = self._prefill_request(req)
                if ar is not None:
                    results.append(ar)
            return results

    def _prefill_request(self, req: _Request) -> Optional[_ActiveRequest]:
        """Prefill a single request and return an ActiveRequest ready for batched decode."""
        try:
            cache = make_prompt_cache(self._model)
            prompt = mx.array(req.prompt_tokens)[None]  # [1, seq_len]
            logits = self._model(prompt, cache=cache)
            mx.eval([c.state for c in cache])

            # Sample first token
            last_logits = logits[:, -1, :]
            if req.temperature <= 0:
                token = mx.argmax(last_logits, axis=-1)
            else:
                # Simple temperature sampling
                token = mx.random.categorical(last_logits / req.temperature)
            mx.eval(token)
            tok_id = token.item()

            # Emit first token
            if tok_id in self._eos_tokens:
                self._emit(req.id, [], done=True)
                return None
            self._emit(req.id, [tok_id], done=False)

            ar = _ActiveRequest(
                id=req.id,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                generated_count=1,
                last_token=tok_id,
                cache=cache,
            )
            return ar
        except Exception as e:
            print(f"[MlxLmEngine] Prefill error: {e}")
            self._emit(req.id, [], done=True)
            return None

    def _batch_decode_step(self, active: list[_ActiveRequest]) -> list[int]:
        """Run one batched decode step for all active requests. Returns indices to retire."""
        B = len(active)
        if B == 0:
            return []

        if B == 1:
            return self._single_decode_step(active)

        # Batched decode: prefill all requests together with [B, 1] input
        # using a SINGLE shared cache (batched from the start).
        # This works when all requests were prefilled together (same prompt length).
        # For heterogeneous prompts, fall back to sequential.
        if not hasattr(active[0], '_batch_cache_ref') or active[0]._batch_cache_ref is None:
            # First time: requests were prefilled individually, process sequentially
            done = []
            for i, ar in enumerate(active):
                result = self._single_decode_step([ar])
                if result:
                    done.append(i)
            return done

        # Use shared batch cache
        input_tokens = mx.array([[ar.last_token] for ar in active])  # [B, 1]
        batch_cache = active[0]._batch_cache_ref

        logits = self._model(input_tokens, cache=batch_cache)  # [B, 1, vocab]
        mx.eval(logits)

        done_indices = []
        for b, ar in enumerate(active):
            req_logits = logits[b:b+1, -1, :]
            if ar.temperature <= 0:
                token = mx.argmax(req_logits, axis=-1)
            else:
                token = mx.random.categorical(req_logits / ar.temperature)
            mx.eval(token)
            tok_id = token.item()

            ar.last_token = tok_id
            ar.generated_count += 1

            if tok_id in self._eos_tokens or ar.generated_count >= ar.max_tokens:
                if tok_id not in self._eos_tokens:
                    self._emit(ar.id, [tok_id], done=False)
                done_indices.append(b)
            else:
                self._emit(ar.id, [tok_id], done=False)

        return done_indices

    def _single_decode_step(self, active: list[_ActiveRequest]) -> list[int]:
        """Decode step for a single active request (no cache stacking needed)."""
        ar = active[0]
        input_tokens = mx.array([[ar.last_token]])  # [1, 1]
        logits = self._model(input_tokens, cache=ar.cache)  # [1, 1, vocab]

        last_logits = logits[:, -1, :]
        if ar.temperature <= 0:
            token = mx.argmax(last_logits, axis=-1)
        else:
            token = mx.random.categorical(last_logits / ar.temperature)
        mx.eval(token)
        tok_id = token.item()

        ar.last_token = tok_id
        ar.generated_count += 1

        if tok_id in self._eos_tokens or ar.generated_count >= ar.max_tokens:
            if tok_id not in self._eos_tokens:
                self._emit(ar.id, [tok_id], done=False)
            return [0]
        else:
            self._emit(ar.id, [tok_id], done=False)
            return []

    # ---------------------------------------------------------------
    # Shared helpers
    # ---------------------------------------------------------------

    def _emit(self, request_id: str, tokens: list[int], done: bool):
        out = TokenOutput(request_id=request_id, tokens=tokens, done=done)
        with self._output_lock:
            self._output_queue.append(out)
