import logging
from dataclasses import dataclass
import egg.core
from datetime import datetime


@dataclass
class ExperimentOptions:
    experiment: str
    embedding_size: int = 50
    hidden_size: int = 20
    game_size: int = 2
    vocab_size: int = 60
    tau_s: float = 10.0
    mixed_distractor_selection: bool = False

    batch_size: int = 64
    batches_per_epoch: int = 16
    n_epochs: int = 50

    dataset_location: str = 'assets/output'
    feat_size: int = 50
    preembed: bool = True
    show_plot: bool = False
    n_separated_shapes: int = 8
    n_unseen_shapes: int = 1
    print_to_console: bool = True

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in vars(d).items() if k in cls.__dataclass_fields__})

    def __post_init__(self):
        self.timestamp = datetime.now().strftime("%Y_%d_%m_%H_%M_%S__")
        out_options = egg.core.init()
        out_options.batch_size = self.batch_size
        out_options.n_epochs = self.n_epochs
        out_options.vocab_size = self.vocab_size

        for k, v in vars(out_options).items():
            if k not in self.__dataclass_fields__:
                setattr(self, k, v)

        if self.embedding_size % 5 != 0:
            self.embedding_size = self.embedding_size + (5 - self.embedding_size % 5)
            logging.warning(f"embedding_size must be a multiple of 5, setting to {self.embedding_size}")

        if self.preembed and self.feat_size != self.embedding_size:
            logging.warning(f"setting feat_size to embedding_size ({self.embedding_size}) (reason: preembed = True)")
            self.feat_size = self.embedding_size

        if not 0 <= self.n_unseen_shapes <= 2:
            raise ValueError(f"n_unseen_shapes_per_test_element must be 0, 1 or 2, got {self.n_unseen_shapes}")

    def __str__(self):
        return f"{self.timestamp}_{self.experiment}_game{self.game_size}_vocab{self.vocab_size}_hidden{self.hidden_size}_unseenShapes{self.n_unseen_shapes}_epochs{self.n_epochs}"
