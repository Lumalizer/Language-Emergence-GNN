import logging
from data.get_loaders import ExtendedDataLoader, get_dataloaders
from analysis.analyze_experiment import get_experiment_means, results_to_dataframe
from experiment.game import get_game
from options import Options
from experiment.train import perform_training
import pandas as pd
from datetime import datetime
import egg.core as core
import os
import torch
import random
import numpy as np
import wandb

def ensure_determinism():
    torch.backends.cudnn.deterministic = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def evalute_model(model, loader: ExtendedDataLoader):
    options = loader.options
    options.eval = True

    target_labels = []
    distractor_labels = []

    def parse_message(message: list[list[float]]):
        return [vocab[word_probs.index(1.0)] for word_probs in message]
    
    def collect_labels(target: str, distractors: list[str]):
        target_labels.append(target)
        distractor_labels.append(distractors) 

    loader.collect_labels = collect_labels
    loss, interaction = model.eval(loader)
    interaction: core.Interaction

    vocab = {i: i for i in range(options.vocab_size)}
    message = [parse_message(m) for m in interaction.message.tolist()]
    accuracies = interaction.aux['acc'].tolist()

    options.eval = False
    return pd.DataFrame({'target': target_labels, 'distractors': distractor_labels, 
                         'message': message, 'accuracy': accuracies}), interaction

def run_experiment(options: Options):
    ensure_determinism()

    logging.info(f"Running {options}")

    wandb.init(project="testing-egg", config=options.to_dict())
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    train_loader, valid_loader = get_dataloaders(options)
    game = get_game(options)

    results, model = perform_training(options, train_loader, valid_loader, game)
    eval_train, interaction_train = evalute_model(model, train_loader)
    eval_test, interaction_test = evalute_model(model, valid_loader)

    wandb.finish()

    return results_to_dataframe(options, results, eval_train, eval_test)


def run_experiments(options: Options, n_repetitions: int = 1):
    run_graph = options.experiment in ['both', 'graph']
    run_image = options.experiment in ['both', 'image']

    graph_options = Options.from_dict(options)
    graph_options.experiment = 'graph'
    image_options = Options.from_dict(options)
    image_options.experiment = 'image'

    results_graph = []
    results_img = []
    for i in range(n_repetitions):
        if n_repetitions > 1: 
            logging.info(f"Running repetition {i+1}/{n_repetitions}")
        run_graph and results_graph.append(run_experiment(graph_options))
        run_image and results_img.append(run_experiment(image_options))

    results_graph = get_experiment_means(results_graph)
    results_img = get_experiment_means(results_img)

    return pd.concat((results_graph, results_img))

def run_series_experiments(experiments: list[Options], name: str, n_repetitions: int = 1):
    results = pd.DataFrame()
    now = datetime.now().strftime("%Y_%d_%m_%H_%M_%S")
    target_folder = f"results/{now + name}"

    for i, options in enumerate(experiments):
        logging.info(f"Running experiment {i+1}/{len(experiments)}")
        options._target_folder = target_folder
        os.makedirs(target_folder+"/experiments", exist_ok=True)
        results = pd.concat((results, run_experiments(options, n_repetitions)))

    results.to_csv(f"{target_folder}/results.csv")
    return results, target_folder
