from dataclasses import dataclass, field
import egg.core
from datetime import datetime
import torch


@dataclass
class Options:
    experiment: str
    name: str = ""
    project_name: str = None

    _target_folder: str = ""
    _timestamp: str = ""
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size: int = 32
    batches_per_epoch: int = 32
    n_epochs: int = 80

    game_size: int = 2
    max_len: int = 4
    vocab_size: int = 20
    embedding_size: int = 30
    hidden_size: int = 80
    image_size: int = 120

    sender_target_only: bool = False
    systemic_distractors: bool = False

    sender_cell: str = 'gru'  # 'rnn', 'gru', or 'lstm'
    length_cost: float = 0.0
    tau_s: float = 1.0

    n_separated_shapes: int = 8
    n_unseen_shapes: int = 0

    print_analysis: bool = False
    print_progress: bool = True

    enable_analysis: bool = True

    results: dict = field(default_factory=dict)
    _eval: bool = False

    @property
    def timestamp(self):
        if self._timestamp == "":
            self._timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        return self._timestamp

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in vars(d).items() if k in cls.__dataclass_fields__})
    
    def to_dict(self):
        return {k: v for k, v in vars(self).items() if k in self.__dataclass_fields__
                and not k in ['_timestamp', '_target_folder', 'eval', 'device', 'enable_analysis', 
                              'print_analysis', 'print_progress', 'n_separated_shapes', 'results']}

    def __post_init__(self):
        out_options = egg.core.init(params=['--random_seed=42',
                                            '--lr=1e-3',
                                            f'--batch_size={self.batch_size}',
                                            f'--n_epochs={self.n_epochs}',
                                            f'--vocab_size={self.vocab_size}'])

        for k, v in vars(out_options).items():
            if k not in self.__dataclass_fields__:
                setattr(self, k, v)

        if not 0 <= self.n_unseen_shapes <= 2:
            raise ValueError(f"n_unseen_shapes_per_test_element must be 0, 1 or 2, got {self.n_unseen_shapes}")

    def __str__(self):
        return f"{self.timestamp + self.name + '_' if self.name else ''}{self.experiment}_maxlen_{self.max_len}_vocab{self.vocab_size}_game{self.game_size}"
