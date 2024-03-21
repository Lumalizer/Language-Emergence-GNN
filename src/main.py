import os
import logging
import coloredlogs
import pandas as pd
from experiment.experiments import run_series_experiments
from analysis.plot import plot_dataframe
from options import Options

logging.basicConfig(level=logging.INFO)
coloredlogs.install(level='INFO')

experiments = [
    Options(experiment='both', game_size=5, max_len=4, vocab_size=7, n_epochs=200, sender_target_only=True, systemic_distractors=False),
    # Options(experiment='both', game_size=5, max_len=4, vocab_size=12, n_epochs=1200, sender_target_only=True, systemic_distractors=True),
]

results, target_folder = run_series_experiments(experiments, 'systemic_', n_repetitions=1)
plot_dataframe(results, "Graph vs Image representations", mode="both", facet_col='mode', facet_row='game_size', save_target=target_folder)
