from typing import Dict, Self


class Vocab:
    UNK_TOKEN = {"id": 0, "token": "#"}

    def __init__(self, token_to_id: Dict[str, int], token_freq: Dict[str, int]) -> None:
        self.token_freq = token_freq
        self.token_to_id = token_to_id
        self.id_to_token = {value: key for key, value in token_to_id.items()}

    @classmethod
    def init_from_text(cls, text: str) -> Self:
        token_to_id = {Vocab.UNK_TOKEN["token"]: Vocab.UNK_TOKEN["id"]}
        token_freq = {}

        for token in text:
            if token not in token_to_id:
                token_id = len(token_to_id)
                token_to_id[token] = token_id

            if token not in token_freq:
                token_freq[token] = 0

            token_freq[token] += 1

        token_freq = dict(sorted(token_freq.items(), key=lambda item: item[1], reverse=True))
        return cls(token_to_id, token_freq)

    def __len__(self) -> int:
        return len(self.token_to_id)
