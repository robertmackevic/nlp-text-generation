from argparse import Namespace

from torch import Tensor
from torch.nn import Module, Embedding, Linear, LSTM as _LSTM


class LSTM(Module):
    def __init__(self, config: Namespace, vocab_size: int) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, config.embedding_dim)
        self.lstm = _LSTM(config.embedding_dim, config.hidden_dim, config.num_layers, batch_first=True)
        self.fc = Linear(config.hidden_dim, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        # [batch_size, seq_length]
        x = self.embedding(x)
        # [batch_size, seq_length, embedding_dim]
        x, _ = self.lstm(x)
        # [batch_size, seq_length, hidden_dim]
        x = self.fc(x[:, -1, :])
        # [batch_size, vocab_size]
        return x
