import re
from argparse import Namespace
from typing import Tuple, Optional

import nltk
from nltk.corpus import gutenberg
from torch import Tensor
from torch.utils.data import Dataset, Subset, DataLoader

from src.data.tokenizer import Tokenizer


class TextGenerationDataset(Dataset):
    def __init__(self, config: Namespace, tokenizer: Optional[Tokenizer] = None) -> None:
        nltk.download("gutenberg")

        self.config = config
        available_works = gutenberg.fileids()

        if not config.data in available_works:
            raise ValueError(f"{config.data} work is not available in the Gutenberg corpus: {available_works}")

        self.text = self.preprocess_text(gutenberg.raw(config.data))
        self.window_size = config.context_length + config.output_length

        self.tokenizer = Tokenizer.init_from_text(self.text) if tokenizer is None else tokenizer

    def __len__(self) -> int:
        return int((len(self.text) - self.window_size) / self.config.window_step)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        start_index = index * self.config.window_step
        end_index = start_index + self.window_size
        sample = self.tokenizer.encode(self.text[start_index:end_index])
        return sample[:self.config.context_length], sample[-self.config.output_length:]

    @staticmethod
    def preprocess_text(text: str) -> str:
        strings = text.split("\n")
        text = "\n".join(
            re.sub(r"[ \t]+", " ", string.lower().strip())
            for string in strings
            if len(string) > 1 and "Book" not in string and not any(char in "0123456789[]()*$\\" for char in string)
        )
        return text

    def split(self, test_fraction: float) -> Tuple[Subset, Subset]:
        total_size = len(self)
        test_size = int(total_size * test_fraction)
        train_size = total_size - test_size
        train_subset = Subset(self, list(range(train_size)))
        test_subset = Subset(self, list(range(train_size, total_size)))
        return train_subset, test_subset


def get_dataloaders(dataset: TextGenerationDataset) -> Tuple[DataLoader, DataLoader]:
    train_subset, test_subset = dataset.split(dataset.config.test_fraction)
    train_dl = DataLoader(train_subset, batch_size=dataset.config.batch_size, shuffle=True, pin_memory=True)
    test_dl = DataLoader(test_subset, batch_size=dataset.config.batch_size, shuffle=True, pin_memory=True)
    return train_dl, test_dl
