import wandb
from dotenv import load_dotenv
import logging
import coloredlogs
from experiment.experiments import Experiment, ExperimentGroup
from options import Options

load_dotenv()
wandb.login()
logging.basicConfig(level=logging.INFO)
coloredlogs.install(level='INFO')

experiment = ExperimentGroup('testing-egg', [
    Experiment(Options(experiment='both', game_size=5, max_len=6, vocab_size=55, n_epochs=100, sender_target_only=True, systemic_distractors=False, enable_analysis=False)),
])

experiment.run()
experiment.plot_dataframe()
