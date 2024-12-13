import json
from pathlib import Path
from typing import Self

import torch

from src.data.vocab import Vocab
from src.paths import TOKENIZER_FILE


class Tokenizer:
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    @classmethod
    def init_from_text(cls, text: str) -> Self:
        return cls(Vocab.init_from_text(text))

    @classmethod
    def init_from_file(cls, filepath: Path = TOKENIZER_FILE) -> Self:
        with open(filepath, "r") as file:
            tokenizer_data = json.load(file)
            return cls(Vocab(tokenizer_data["token_to_id"], tokenizer_data["token_freq"]))

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.vocab.token_to_id.get(token, Vocab.UNK_TOKEN["id"]) for token in text])

    def decode(self, tensor: torch.Tensor) -> str:
        return "".join(self.vocab.id_to_token.get(token_id, Vocab.UNK_TOKEN["token"]) for token_id in tensor.tolist())

    def compute_class_weights(self) -> torch.Tensor:
        total_tokens = sum(self.vocab.token_freq.values())
        weights = torch.tensor([
            0 if freq == 0 else total_tokens / freq for freq in self.vocab.token_freq.values()
        ])
        return weights / weights.max()

    def save(self, filepath: Path) -> None:
        with open(filepath, "w") as file:
            json.dump({
                "token_to_id": self.vocab.token_to_id,
                "token_freq": self.vocab.token_freq,
            }, file, indent=4)  # type: ignore
