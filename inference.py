from argparse import ArgumentParser, Namespace

import torch
from torch.nn.functional import softmax

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
    parser.add_argument("--temperature", type=float, required=False, default=0.0)
    return parser.parse_args()


def run(version: str, weights: str, seed_sequence: str, max_tokens: int, temperature: float) -> None:
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

    text = seed_sequence.replace(r"\n", "\n")
    for _ in range(max_tokens):
        source = tokenizer.encode(text[-config.context_length:]).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(source)

        if temperature == 0.0:
            prediction = torch.argmax(output, dim=1)

        else:
            logits = output[0, :]
            logits = logits / temperature
            probabilities = softmax(logits, dim=-1)
            prediction = torch.multinomial(probabilities, num_samples=1)

        text += tokenizer.decode(prediction)

    logger.info("\n" + text)
    logger.info(f"Length: {len(text)}")


if __name__ == "__main__":
    run(**vars(parse_args()))
