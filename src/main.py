import os
import logging
import coloredlogs
import pandas as pd
from experiment.experiments import run_series_experiments
from analysis.plot import plot_dataframe
from experiments import experiments

logging.basicConfig(level=logging.INFO)
coloredlogs.install(level='INFO')

# Set working directory to the directory of this file
os.chdir(os.path.dirname(os.path.realpath(__file__)))

results, target_folder = run_series_experiments(experiments, 'graphvsimage', n_repetitions=1)
#results = pd.read_csv('../results/2023_17_09_23_04_54graphvsimage/results.csv')

# plot_dataframe(results, "Train", mode="train", facet_col='game_size', facet_row='game_size', save_target=target_folder)
# plot_dataframe(results, "Test", mode="test", facet_col='game_size', facet_row='game_size', save_target=target_folder)
plot_dataframe(results, "Graph vs Image representations", mode="both", facet_col='mode', facet_row='game_size', save_target=target_folder)
