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

results = run_series_experiments(experiments, '3graphnodesmulti', n_repetitions=1)
#results = pd.read_csv('../results/2023_23_06_02_30_093graphnodesmulti/results.csv')

plot_dataframe(results, "Test", mode="test", facet_col='max_len', facet_row='game_size')
