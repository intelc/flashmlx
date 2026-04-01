"""Entry point for the FlashMLX HTTP server.

Usage:
    python -m server.run <model_path> [--host HOST] [--port PORT]
                         [--max-batch-size N] [--max-context-len N]
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="FlashMLX HTTP server")
    parser.add_argument("model_path", help="Path to the model directory")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Max batch size (default: 8)")
    parser.add_argument("--max-context-len", type=int, default=2048, help="Max context length (default: 2048)")
    args = parser.parse_args()

    # Import here so the module is available before uvicorn forks
    from server.python.app import app, init

    init(args.model_path, args.max_batch_size, args.max_context_len)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
