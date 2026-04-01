from transformers import AutoTokenizer, PreTrainedTokenizerFast
from pathlib import Path
import json


class Tokenizer:
    def __init__(self, model_path: str):
        try:
            self.tok = AutoTokenizer.from_pretrained(model_path)
        except (KeyError, ValueError, TypeError):
            # Fallback for models with config parsing issues (e.g., nemotron_h)
            # Load tokenizer directly from tokenizer.json without AutoConfig
            tok_json = Path(model_path) / "tokenizer.json"
            if tok_json.exists():
                self.tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_json))
                # Try to load special tokens
                tok_config = Path(model_path) / "tokenizer_config.json"
                if tok_config.exists():
                    with open(tok_config) as f:
                        tc = json.load(f)
                    if "chat_template" in tc:
                        self.tok.chat_template = tc["chat_template"]
            else:
                raise RuntimeError(f"Cannot load tokenizer from {model_path}")

    def encode_chat(self, messages: list[dict]) -> list[int]:
        try:
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return self.tok.encode(text)
        except (ValueError, KeyError):
            # Fallback for base models without chat template
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            text += "\nassistant:"
            return self.tok.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tok.decode(token_ids, skip_special_tokens=True)

    def decode_token(self, token_id: int) -> str:
        return self.tok.decode([token_id], skip_special_tokens=False)
