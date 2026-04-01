from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, model_path: str):
        self.tok = AutoTokenizer.from_pretrained(model_path)

    def encode_chat(self, messages: list[dict]) -> list[int]:
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tok.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tok.decode(token_ids, skip_special_tokens=True)

    def decode_token(self, token_id: int) -> str:
        return self.tok.decode([token_id], skip_special_tokens=False)
