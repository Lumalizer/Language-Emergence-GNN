import json
import os
import uuid
import pandas as pd
import statistics
from options import ExperimentOptions


def get_experiment_means(results: list[pd.DataFrame]):
    to_mean = ['acc', 'loss', 'baseline', 'sender_entropy', 'receiver_entropy']
    r = results[0]
    for element in to_mean:
        try:
            target = [r[element] for r in results]
            means = [statistics.mean(target) for target in zip(*target)]
            r[element] = means
        except KeyError:
            pass
    return r


def results_to_dataframe(results: str, options: ExperimentOptions, folder: str, save: bool = True) -> pd.DataFrame:
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
    results['max_len'] = int(options.max_len)
    results['sender_cell'] = str(options.sender_cell)
    results['id'] = str(uuid.uuid4())
    os.makedirs(f'{folder}/small', exist_ok=True)
    save and results.to_csv(f'{folder}/small/{str(options)}.csv')
    return results


def get_final_accuracies(filename: str):
    filename = os.path.abspath(filename)
    df = pd.read_csv(filename)
    mask = df['epoch'] == 300
    df = df[mask]
    return df[['experiment', 'mode', 'acc', 'max_len', 'game_size']].reset_index(drop=True)
