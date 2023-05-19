import json
import logging
import uuid
from data.data_generation import assure_dataset
from data.get_loaders import get_dataloaders
from game import get_game
from options import ExperimentOptions
from train import perform_training
import pandas as pd
import statistics
from datetime import datetime
import os


def results_to_dataframe(results: str, options: ExperimentOptions, folder: str = '', save: bool = True) -> pd.DataFrame:
    initial = pd.DataFrame({'experiment': options.experiment, 'mode': 'train', 'epoch': [0], 'acc': [1/options.game_size]})
    initial = pd.concat((initial, pd.DataFrame({'experiment': options.experiment, 'mode': 'test', 'epoch': [0], 'acc': [1/options.game_size]})))
    results = pd.concat((initial, pd.DataFrame([json.loads(line) for line in results.split('\n') if line])))
    results['experiment'] = str(options.experiment)
    results['n_unseen_shapes'] = int(options.n_unseen_shapes)
    results['game_size'] = int(options.game_size)
    results['vocab_size'] = int(options.vocab_size)
    results['hidden_size'] = int(options.hidden_size)
    results['n_epochs'] = int(options.n_epochs)
    results['embedding_size'] = int(options.embedding_size)
    results['batch_size'] = int(options.batch_size)
    results['id'] = str(uuid.uuid4())
    not os.path.exists(f'results/{folder}') and os.makedirs(f'results/{folder}')
    save and results.to_csv(f'results/{folder + str(options)}.csv')
    return results


def run_experiment(options: ExperimentOptions, folder: str = 'small/'):
    assure_dataset()
    train_loader, valid_loader = get_dataloaders(options)
    game = get_game(options)
    results = perform_training(options, train_loader, valid_loader, game)
    return results_to_dataframe(results, options, folder)


def get_experiment_means(results: list[pd.DataFrame]):
    to_mean = ['acc', 'loss', 'baseline', 'sender_entropy', 'receiver_entropy']
    r = results[0]
    for element in to_mean:
        target = [r[element] for r in results]
        means = [statistics.mean(target) for target in zip(*target)]
        r[element] = means
    return r


def run_both_experiments(options: ExperimentOptions, n_repetitions: int = 1):
    assert options.experiment == 'both'
    graph_options = ExperimentOptions.from_dict(options)
    graph_options.experiment = 'graph'
    image_options = ExperimentOptions.from_dict(options)
    image_options.experiment = 'image'

    results_graph = []
    results_img = []

    for i in range(n_repetitions):
        logging.info(f"Running repetition {i+1}/{n_repetitions}")
        results_graph.append(run_experiment(graph_options))
        results_img.append(run_experiment(image_options))

    results_graph = get_experiment_means(results_graph)
    results_img = get_experiment_means(results_img)

    return pd.concat((results_graph, results_img))


def run_series_experiments(experiments: list[ExperimentOptions], name: str, n_repetitions: int = 1):
    results = pd.DataFrame()
    now = datetime.now().strftime("_%Y_%d_%m_%H_%M_%S")
    for i, options in enumerate(experiments):
        logging.info(f"Running experiment {i+1}/{len(experiments)}")
        results = pd.concat((results, run_both_experiments(options, n_repetitions)))
    results.to_csv(f"results/{name + now}.csv")
    return results
