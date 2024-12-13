from torch.utils.data import DataLoader

from src.data.dataset import TextGenerationDataset
from src.trainer import Trainer
from src.utils import get_logger, load_config, seed_everything, count_parameters


def run() -> None:
    logger = get_logger()
    config = load_config()
    seed_everything(config.seed)

    logger.info("Preparing the data...")
    dataset = TextGenerationDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)

    vocab_size = len(dataset.tokenizer.vocab)
    trainer = Trainer(config, vocab_size)
    logger.info(f"Number of trainable parameters: {count_parameters(trainer.model):,}")
    logger.info(f"Vocabulary size: {vocab_size:,}")

    try:
        logger.info("Starting training...")
        trainer.fit(dataloader)
    except KeyboardInterrupt:
        logger.info("Training terminated.")


if __name__ == "__main__":
    run()
