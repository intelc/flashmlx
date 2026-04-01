import asyncio
import time
import uuid
from typing import AsyncGenerator, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# These are populated by init()
engine = None
tokenizer = None
pending_queues: dict[str, asyncio.Queue] = {}

app = FastAPI(title="FlashMLX", version="0.1.0")


# ── Pydantic models ───────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = 256
    temperature: float = 1.0
    stream: bool = False
    model: str = "flashmlx"


# ── Startup / background ──────────────────────────────────────────────────────

def init(model_path: str, max_batch_size: int = 8, max_context_len: int = 2048):
    """Called before uvicorn starts serving. Loads model + tokenizer."""
    global engine, tokenizer

    from server.python._flashmlx_engine import Engine
    from server.python.tokenizer import Tokenizer

    tokenizer = Tokenizer(model_path)
    engine = Engine(model_path, max_batch_size, max_context_len)
    engine.start()


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(poll_loop())


async def poll_loop():
    """Background task: drain C++ engine and route tokens to per-request queues."""
    while True:
        if engine is not None:
            outputs = engine.poll_tokens()
            for out in outputs:
                q = pending_queues.get(out.request_id)
                if q is not None:
                    await q.put(out)
        await asyncio.sleep(0.005)


@app.on_event("shutdown")
async def shutdown_event():
    if engine is not None:
        engine.stop()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_request_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _sse_chunk(request_id: str, model: str, delta_content: str, finish_reason: Optional[str] = None) -> str:
    import json
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta_content} if delta_content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine_ready": engine is not None,
        "tokenizer_ready": tokenizer is not None,
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    request_id = _make_request_id()
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt_token_ids = tokenizer.encode_chat(messages)
    prompt_tokens = len(prompt_token_ids)

    q: asyncio.Queue = asyncio.Queue()
    pending_queues[request_id] = q

    engine.submit_request(request_id, prompt_token_ids, req.max_tokens, req.temperature)

    if req.stream:
        return StreamingResponse(
            _stream_tokens(request_id, req.model, q),
            media_type="text/event-stream",
        )
    else:
        return await _collect_response(request_id, req.model, q, prompt_tokens)


async def _stream_tokens(request_id: str, model: str, q: asyncio.Queue) -> AsyncGenerator[str, None]:
    """Yield SSE chunks for each token until done."""
    try:
        # Send initial role delta
        import json
        role_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"

        while True:
            out = await q.get()
            for token_id in out.tokens:
                text = tokenizer.decode_token(token_id)
                yield _sse_chunk(request_id, model, text)
            if out.done:
                yield _sse_chunk(request_id, model, "", finish_reason="stop")
                yield "data: [DONE]\n\n"
                break
    finally:
        pending_queues.pop(request_id, None)


async def _collect_response(request_id: str, model: str, q: asyncio.Queue, prompt_tokens: int) -> dict:
    """Collect all tokens then return a complete chat completion object."""
    try:
        all_token_ids: list[int] = []
        while True:
            out = await q.get()
            all_token_ids.extend(out.tokens)
            if out.done:
                break

        content = tokenizer.decode(all_token_ids)
        completion_tokens = len(all_token_ids)

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    finally:
        pending_queues.pop(request_id, None)
