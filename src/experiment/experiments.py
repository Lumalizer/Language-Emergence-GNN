from dataclasses import dataclass, field
import logging
from data.get_loaders import ExtendedDataLoader, get_dataloaders
from analysis.analyze_experiment import results_to_dataframe
from analysis.plot import plot_dataframe
from experiment.language_game import get_game
from options import Options
from experiment.language_game import perform_training
import pandas as pd
from datetime import datetime
import egg.core as core
import os
import torch
import random
import numpy as np
import wandb
from copy import deepcopy

@dataclass
class Experiment:
    options: Options
    model: core.Trainer = None
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    eval_train: pd.DataFrame = field(default_factory=pd.DataFrame)
    eval_test: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_loader: ExtendedDataLoader = None
    valid_loader: ExtendedDataLoader = None


    @staticmethod
    def ensure_determinism():
        torch.backends.cudnn.deterministic = True
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def run(self):
        self.ensure_determinism()

        options = self.options

        self.train_loader, self.valid_loader = get_dataloaders(self.options)
        self.game = get_game(self.options)

        wandb.init(project=self.options.project_name, config=options.to_dict())
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

        results, self.model = perform_training(options, self.train_loader, self.valid_loader, self.game)
        self.eval_train, interaction_train = self.evaluate(self.train_loader)
        self.eval_test, interaction_test = self.evaluate(self.valid_loader)

        wandb.finish()

        self.results = results_to_dataframe(options, results, self.eval_train, self.eval_test)
        return self.results

    def evaluate(self, loader: ExtendedDataLoader):
        options = loader.options
        options._eval = True

        target_labels = []
        distractor_labels = []

        def parse_message(message: list[list[float]]):
            return [vocab[word_probs.index(1.0)] for word_probs in message]
        
        def collect_labels(target: str, distractors: list[str]):
            target_labels.append(target)
            distractor_labels.append(distractors) 

        loader.collect_labels = collect_labels
        with torch.no_grad():
            loss, interaction = self.model.eval(loader)
        interaction: core.Interaction

        vocab = {i: i for i in range(options.vocab_size)}
        message = [parse_message(m) for m in interaction.message.tolist()]
        accuracies = interaction.aux['acc'].tolist()

        options._eval = False
        return pd.DataFrame({'target': target_labels, 'distractors': distractor_labels, 
                            'message': message, 'accuracy': accuracies}), interaction


@dataclass
class ExperimentGroup:
    name: str
    experiments: list[Experiment]
    results: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_folder: str = None

    def __post_init__(self):
        experiments = []
        for experiment in self.experiments:
            if experiment.options.experiment == 'both':
                e1, e2 = deepcopy(experiment), deepcopy(experiment)
                e1.options.experiment = 'graph'
                e2.options.experiment = 'image'
                experiments.extend([e1, e2])
            else:
                experiments.append(experiment)

        self.experiments = experiments

    def run(self):
        now = datetime.now().strftime("%Y_%d_%m_%H_%M_%S")
        self.target_folder = f"results/{now + self.name}"

        for i, experiment in enumerate(self.experiments):
            logging.info(f"Running experiment {i+1}/{len(self.experiments)} :: {experiment.options}")
            experiment.options._target_folder = self.target_folder
            os.makedirs(self.target_folder+"/experiments", exist_ok=True)
            experiment.options.project_name = self.name
            experiment.run()

        self.results = pd.concat([e.results for e in self.experiments])
        self.results.to_csv(f"{self.target_folder}/results.csv")

    def plot_dataframe(self, name=None, mode="both", facet_col='mode', facet_row='game_size'):
        plot_dataframe(self.results, name if name else self.name, mode=mode, facet_col=facet_col, facet_row=facet_row, save_target=self.target_folder)
