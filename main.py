import os
import coloredlogs
import logging
from options import ExperimentOptions
from experiment import run_series_experiments
from utils.plot import plot_test_data, plot_train_data
import pandas as pd

logging.basicConfig(level=logging.INFO)
coloredlogs.install(level='INFO')

# Set working directory to the directory of this file
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# example of running a series of experiments
experiments = [
    ExperimentOptions(experiment='both', game_size=2, n_unseen_shapes=0),
    ExperimentOptions(experiment='both', game_size=2, n_unseen_shapes=1),
    ExperimentOptions(experiment='both', game_size=2, n_unseen_shapes=2),

    ExperimentOptions(experiment='both', game_size=5, n_unseen_shapes=0),
    ExperimentOptions(experiment='both', game_size=5, n_unseen_shapes=1),
    ExperimentOptions(experiment='both', game_size=5, n_unseen_shapes=2),

    ExperimentOptions(experiment='both', game_size=15, n_unseen_shapes=0, vocab_size=60),
    ExperimentOptions(experiment='both', game_size=15, n_unseen_shapes=1, vocab_size=60),
    ExperimentOptions(experiment='both', game_size=15, n_unseen_shapes=2, vocab_size=60)
]

# results = run_series_experiments(experiments, 'example_results', n_repetitions=1)
results = pd.read_csv('results/original_results_mean_10.csv')

plot_test_data(results, vocab_size=10)
# plot_test_data(results, vocab_size=60)
