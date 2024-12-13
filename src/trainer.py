from argparse import Namespace
from os import listdir, makedirs
from typing import Dict, Optional

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from src.lstm import LSTM
from src.paths import RUNS_DIR, CONFIG_FILE, TOKENIZER_FILE
from src.utils import get_available_device, get_logger, save_config, save_weights


class AverageMeter:
    def __init__(self) -> None:
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float) -> None:
        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count


class Trainer:
    def __init__(self, config: Namespace, vocab_size: int) -> None:
        self.config = config
        self.device = get_available_device()
        self.logger = get_logger(__name__)

        self.model = LSTM(config, vocab_size).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.loss_fn = CrossEntropyLoss().to(self.device)

    def fit(self, dataloader: DataLoader) -> None:
        RUNS_DIR.mkdir(exist_ok=True, parents=True)
        model_dir = RUNS_DIR / f"v{len(listdir(RUNS_DIR)) + 1}"

        summary_writer_train = SummaryWriter(log_dir=model_dir / "train")

        makedirs(summary_writer_train.log_dir, exist_ok=True)
        save_config(self.config, model_dir / CONFIG_FILE.name)
        dataloader.dataset.tokenizer.save(model_dir / TOKENIZER_FILE.name)  # type: ignore

        for epoch in range(1, self.config.epochs + 1):
            self.logger.info(f"[Epoch {epoch} / {self.config.epochs}]")
            metrics = self._train_for_epoch(dataloader)
            self.log_metrics(metrics, summary_writer_train, epoch=epoch)

            if epoch % self.config.save_interval == 0:
                self.logger.info(f"Saving model weights at epoch: {epoch}")
                save_weights(model_dir / f"weights_{epoch}.pth", self.model)

    def _train_for_epoch(self, dataloader: DataLoader) -> Dict[str, AverageMeter]:
        self.model.train()
        metrics = {
            "loss": AverageMeter(),
        }

        for batch in tqdm(dataloader):
            self.optimizer.zero_grad()
            source = batch[0].to(self.device)
            target = batch[1].to(self.device).squeeze()
            output = self.model(source)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            metrics["loss"].update(loss.item())

        return metrics

    def log_metrics(
            self,
            metrics: Dict[str, AverageMeter],
            summary_writer: Optional[SummaryWriter] = None,
            epoch: Optional[int] = None
    ) -> None:
        message = "\n"
        for metric, value in metrics.items():
            message += f"\t{metric}: {value.avg:.3f}\n"

            if epoch is not None and summary_writer is not None:
                summary_writer.add_scalar(tag=metric, scalar_value=value.avg, global_step=epoch)

        self.logger.info(message)
