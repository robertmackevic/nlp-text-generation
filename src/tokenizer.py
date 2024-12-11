import json
from pathlib import Path
from typing import Self

import torch
from torch.nn.functional import one_hot

from src.paths import TOKENIZER_FILE
from src.vocab import Vocab


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
        token_ids = [self.vocab.token_to_id.get(token, Vocab.UNK_TOKEN["id"]) for token in text]
        return one_hot(torch.tensor(token_ids), num_classes=len(self.vocab)).float()

    def decode(self, one_hot_tensor: torch.Tensor) -> str:
        token_ids = torch.argmax(one_hot_tensor, dim=1).tolist()
        return "".join(self.vocab.id_to_token.get(token_id, Vocab.UNK_TOKEN["token"]) for token_id in token_ids)

    def save(self, filepath: Path) -> None:
        with open(filepath, "w") as file:
            json.dump({
                "token_to_id": self.vocab.token_to_id,
                "token_freq": self.vocab.token_freq,
            }, file, indent=4)
