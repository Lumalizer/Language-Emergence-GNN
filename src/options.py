import logging
from dataclasses import dataclass
import egg.core
from datetime import datetime
import torch


@dataclass
class ExperimentOptions:
    experiment: str
    game_size: int = 2
    max_len: int = 4
    vocab_size: int = 20

    embedding_size: int = 30
    hidden_size: int = 80
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    sender_target_only: bool = False

    sender_cell: str = 'gru'  # 'rnn', 'gru', or 'lstm'
    length_cost: float = 0.0
    tau_s: float = 1.0

    batch_size: int = 32
    batches_per_epoch: int = 32
    n_epochs: int = 80

    n_separated_shapes: int = 8
    n_unseen_shapes: int = 0

    dataset_location: str = 'assets/output'
    print_to_console: bool = True
    use_mixed_distractors: bool = False
    use_systematic_distractors: bool = False

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in vars(d).items() if k in cls.__dataclass_fields__})

    def __post_init__(self):
        self.timestamp = datetime.now().strftime("%Y_%d_%m_%H_%M_%S__")
        out_options = egg.core.init(params=['--random_seed=42',
                                            '--lr=1e-3',
                                            f'--batch_size={self.batch_size}',
                                            f'--n_epochs={self.n_epochs}',
                                            f'--vocab_size={self.vocab_size}'])

        for k, v in vars(out_options).items():
            if k not in self.__dataclass_fields__:
                setattr(self, k, v)

        if self.embedding_size % 3 != 0:
            self.embedding_size = self.embedding_size + (3 - self.embedding_size % 3)
            logging.warning(f"embedding_size must be a multiple of 3, setting to {self.embedding_size}")

        if not 0 <= self.n_unseen_shapes <= 2:
            raise ValueError(f"n_unseen_shapes_per_test_element must be 0, 1 or 2, got {self.n_unseen_shapes}")

    def __str__(self):
        return f"{self.experiment}_maxlen_{self.max_len}_vocab{self.vocab_size}_game{self.game_size}"
