from src.data.dataset import TextGenerationDataset, get_dataloaders
from src.trainer import Trainer
from src.utils import get_logger, load_config, seed_everything, count_parameters


def run() -> None:
    logger = get_logger()
    config = load_config()
    seed_everything(config.seed)

    logger.info("Preparing the data...")
    dataset = TextGenerationDataset(config)
    train_dl, test_dl = get_dataloaders(dataset)

    trainer = Trainer(config, dataset.tokenizer)
    logger.info(f"Number of trainable parameters: {count_parameters(trainer.model):,}")
    logger.info(f"Vocabulary size: {len(dataset.tokenizer.vocab):,}")

    try:
        logger.info("Starting training...")
        trainer.fit(train_dl, test_dl)
    except KeyboardInterrupt:
        logger.info("Training terminated.")


if __name__ == "__main__":
    run()
