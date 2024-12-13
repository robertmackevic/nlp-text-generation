from argparse import Namespace
from typing import Tuple

import nltk
from nltk.corpus import gutenberg
from torch import Tensor
from torch.utils.data import Dataset

from src.tokenizer import Tokenizer
from src.utils import nearest_divisible


class TextGenerationDataset(Dataset):
    def __init__(self, config: Namespace) -> None:
        nltk.download("gutenberg")

        self.config = config
        available_works = gutenberg.fileids()

        if not config.data in available_works:
            raise ValueError(f"{config.data} work is not available in the Gutenberg corpus: {available_works}")

        self.text = gutenberg.raw(config.data)
        self.window_size = config.context_length + config.output_length

        self.tokenizer = (
            Tokenizer.init_from_text(self.text)
            if config.tokenizer is None
            else Tokenizer.init_from_file(config.tokenizer)
        )

    def __len__(self) -> int:
        return int(nearest_divisible(len(self.text), self.window_size) / self.config.window_step)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        start_index = index * self.config.window_step
        end_index = start_index + self.window_size
        sample = self.tokenizer.encode(self.text[start_index:end_index])
        return sample[:self.config.context_length], sample[-self.config.output_length:]
