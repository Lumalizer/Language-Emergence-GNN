import logging
from analysis.vocab import store_vocab_info
from data.assure_dataset import assure_dataset
from data.get_loaders import get_dataloaders
from analysis.analyze_experiment import get_experiment_means, results_to_dataframe
from experiment.game import get_game
from options import ExperimentOptions
from experiment.train import perform_training
import pandas as pd
from datetime import datetime
import os


def run_experiment(options: ExperimentOptions, target_folder: str):
    assure_dataset(options)
    print(f"Running {options}")
    target_labels = []

    train_loader, valid_loader = get_dataloaders(options, target_labels)
    game = get_game(options)
    results, model = perform_training(options, train_loader, valid_loader, game)

    target_labels.clear() # only take eval labels
    loss, interaction = model.eval(valid_loader)

    vocab = {i: i for i in range(options.vocab_size)}

    def parse_message(message: list[list[float]]):
        return [vocab[word_probs.index(1.0)] for word_probs in message]

    messages = [parse_message(m) for m in interaction.message.tolist()]

    store_vocab_info(options, target_labels, messages, interaction.aux['acc'].tolist(), target_folder)
    return results_to_dataframe(results, options, target_folder)


def run_experiments(options: ExperimentOptions, target_folder: str, n_repetitions: int = 1):
    run_graph = options.experiment == 'both' or options.experiment == 'graph'
    run_image = options.experiment == 'both' or options.experiment == 'image'

    graph_options = ExperimentOptions.from_dict(options)
    graph_options.experiment = 'graph'
    image_options = ExperimentOptions.from_dict(options)
    image_options.experiment = 'image'

    results_graph = []
    results_img = []

    for i in range(n_repetitions):
        logging.info(f"Running repetition {i+1}/{n_repetitions}")
        run_graph and results_graph.append(run_experiment(graph_options, target_folder))
        run_image and results_img.append(run_experiment(image_options, target_folder))

    if n_repetitions > 1:
        results_graph = get_experiment_means(results_graph)
        results_img = get_experiment_means(results_img)
    else:
        results_graph = results_graph[0] if run_graph else pd.DataFrame()
        results_img = results_img[0] if run_image else pd.DataFrame()

    return pd.concat((results_graph, results_img))


def run_series_experiments(experiments: list[ExperimentOptions], name: str, n_repetitions: int = 1):
    results = pd.DataFrame()
    now = datetime.now().strftime("%Y_%d_%m_%H_%M_%S")
    target_folder = f"../results/{now + name}"
    os.makedirs(target_folder, exist_ok=True)

    for i, options in enumerate(experiments):
        logging.info(f"Running experiment {i+1}/{len(experiments)}")
        results = pd.concat((results, run_experiments(options, target_folder, n_repetitions)))
    results.to_csv(f"{target_folder}/results.csv")
    return results, target_folder
