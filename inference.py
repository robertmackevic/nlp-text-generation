from argparse import ArgumentParser, Namespace

import torch

from src.data.tokenizer import Tokenizer
from src.lstm import LSTM
from src.paths import RUNS_DIR, CONFIG_FILE, TOKENIZER_FILE
from src.utils import get_logger, load_config, seed_everything, load_weights, count_parameters, get_available_device


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-v", "--version", type=str, required=True, help="v1, v2, v3, etc.")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Name of the .pth file")
    parser.add_argument("--seed-sequence", type=str, required=True, help="Text to start the inference")
    parser.add_argument("--max-tokens", type=int, required=True, help="Number of tokens to generate")
    return parser.parse_args()


def run(version: str, weights: str, seed_sequence: str, max_tokens: int) -> None:
    logger = get_logger()
    model_dir = RUNS_DIR / version

    config = load_config(model_dir / CONFIG_FILE.name)
    seed_everything(config.seed)

    if len(seed_sequence) < config.context_length:
        raise ValueError(
            f"Length of seed sequence is less than context length: "
            f"{len(seed_sequence)} < {config.context_length}"
        )

    if max_tokens < 1:
        raise ValueError(f"Max tokens is less than 1")

    tokenizer = Tokenizer.init_from_file(model_dir / TOKENIZER_FILE.name)
    device = get_available_device()

    model = load_weights(
        filepath=model_dir / weights,
        model=LSTM(config, vocab_size=len(tokenizer.vocab))
    ).to(device)
    model.eval()

    logger.info(
        f"\nNumber of parameters: {count_parameters(model):,}"
        f"\nVocabulary size: {len(tokenizer.vocab):,}"
    )

    text = seed_sequence
    for _ in range(max_tokens):
        source = tokenizer.encode(text[-config.context_length:]).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(source)
            prediction = torch.argmax(output, dim=1)

        text += tokenizer.decode(prediction.squeeze())

    logger.info(text)


if __name__ == "__main__":
    run(**vars(parse_args()))
