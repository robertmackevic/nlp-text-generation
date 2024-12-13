from argparse import Namespace, ArgumentParser

from src.data.dataset import TextGenerationDataset, get_dataloaders
from src.paths import RUNS_DIR, CONFIG_FILE
from src.trainer import Trainer
from src.utils import get_logger, load_config, seed_everything, load_weights, count_parameters


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-v", "--version", type=str, required=True, help="v1, v2, v3, etc.")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Name of the .pth file")
    return parser.parse_args()


def run(version: str, weights: str) -> None:
    logger = get_logger()
    model_dir = RUNS_DIR / version

    config = load_config(model_dir / CONFIG_FILE.name)
    seed_everything(config.seed)

    logger.info("Preparing the data...")
    dataset = TextGenerationDataset(config)
    train_dl, test_dl = get_dataloaders(dataset)

    trainer = Trainer(config, dataset.tokenizer)
    trainer.model = load_weights(filepath=model_dir / weights, model=trainer.model)
    logger.info(f"Number of trainable parameters: {count_parameters(trainer.model):,}")
    logger.info(f"Vocabulary size: {len(dataset.tokenizer.vocab):,}")

    try:
        logger.info("Evaluating on training data...")
        trainer.log_metrics(trainer.eval(train_dl))
        trainer.log_metrics(trainer.eval(test_dl))
    except KeyboardInterrupt:
        logger.info("Evaluation terminated.")


if __name__ == "__main__":
    run(**vars(parse_args()))
