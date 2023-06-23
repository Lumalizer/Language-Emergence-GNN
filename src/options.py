import logging
from dataclasses import dataclass
import egg.core
from datetime import datetime


@dataclass
class ExperimentOptions:
    experiment: str
    game_size: int = 2
    max_len: int = 4
    vocab_size: int = 20

    embedding_size: int = 30
    hidden_size: int = 60
    tau_s: float = 1.0
    sender_cell: str = 'gru'  # 'rnn', 'gru', or 'lstm'

    batch_size: int = 64
    batches_per_epoch: int = 16
    n_epochs: int = 300

    feat_size: int = 30
    n_separated_shapes: int = 8
    n_unseen_shapes: int = 1

    dataset_location: str = '../assets/output'
    show_plot: bool = False
    preembedded_data: bool = True
    print_to_console: bool = True
    add_shape_only_games: bool = False

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

        if self.preembedded_data and self.feat_size != self.embedding_size:
            self.feat_size = self.embedding_size

        if not 0 <= self.n_unseen_shapes <= 2:
            raise ValueError(f"n_unseen_shapes_per_test_element must be 0, 1 or 2, got {self.n_unseen_shapes}")

    def __str__(self):
        return f"{self.timestamp}_{self.experiment}_maxlen_{self.max_len}_cell{self.sender_cell}_game{self.game_size}_vocab{self.vocab_size}_hidden{self.hidden_size}_unseen{self.n_unseen_shapes}_epochs{self.n_epochs}"
