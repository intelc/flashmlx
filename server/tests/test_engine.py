import time
import pytest
import sys
sys.path.insert(0, '.')
from server.python._flashmlx_engine import Engine

QWEN_PATH = "/Users/yihengchen/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8"

@pytest.fixture(scope="module")
def engine():
    e = Engine(QWEN_PATH, 4, 512)
    e.start()
    yield e
    e.stop()

_token_buffer = {}  # request_id -> list of tokens
_done_set = set()   # request_ids that are done

def _drain_engine(engine):
    """Drain all pending poll results into the shared buffer."""
    for out in engine.poll_tokens():
        rid = out.request_id
        if rid not in _token_buffer:
            _token_buffer[rid] = []
        if out.done:
            _done_set.add(rid)
        else:
            _token_buffer[rid].extend(out.tokens)

def collect_tokens(engine, request_id, timeout=30):
    """Helper: poll until request is done, return tokens."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        _drain_engine(engine)
        if request_id in _done_set:
            return _token_buffer.get(request_id, [])
        time.sleep(0.01)
    raise TimeoutError(f"Request {request_id} didn't complete in {timeout}s")

class TestEngine:
    def test_ping(self, engine):
        assert engine.ping() == "flashmlx engine ready"

    def test_single_request(self, engine):
        engine.submit_request("test-single", list(range(1, 17)), 10, 0.0)
        tokens = collect_tokens(engine, "test-single")
        assert len(tokens) == 10
        assert all(isinstance(t, int) for t in tokens)
        assert all(0 <= t < 200000 for t in tokens)

    def test_concurrent_requests(self, engine):
        engine.submit_request("a", list(range(1, 9)), 5, 0.0)
        engine.submit_request("b", list(range(1, 9)), 5, 0.0)
        tokens_a = collect_tokens(engine, "a")
        tokens_b = collect_tokens(engine, "b")
        assert len(tokens_a) == 5
        assert len(tokens_b) == 5

    def test_stats(self, engine):
        stats = engine.get_stats()
        assert stats.total_requests >= 3  # from previous tests
