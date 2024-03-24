import json
import os
import uuid
import pandas as pd
import statistics
from options import Options
from analysis.vocab import vocab_error_analysis


def get_experiment_means(results: list[pd.DataFrame]):
    if not results:
        return pd.DataFrame()
    elif len(results) == 1:
        return results[0]

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


def results_to_dataframe(options: Options, results: str, eval_train: pd.DataFrame, eval_test: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    folder = options._target_folder
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
    os.makedirs(f'{folder}/experiments', exist_ok=True)
    save and results.to_csv(f'{folder}/experiments/{str(options)}.csv')

    with open(f'{folder}/experiments/{"vocab_train_" + str(options)}.json', 'w') as f:
        eval_train.to_json(f)
    with open(f'{folder}/experiments/{"vocab_test_" + str(options)}.json', 'w') as f:
        eval_test.to_json(f)

    # with open(f'{folder}/experiments/{"interaction_" + str(options)}.pkl', 'wb') as f:
    #     pickle.dump(interaction, f)        

    with open(f'{folder}/experiments/{"vocab_info_train_" + str(options)}.txt', 'w') as f:
        error_analyis = vocab_error_analysis(eval_train)
        options.print_analysis and print(error_analyis)
        f.write(error_analyis)

    with open(f'{folder}/experiments/{"vocab_info_test_" + str(options)}.txt', 'w') as f:
        error_analyis = vocab_error_analysis(eval_test)
        options.print_analysis and print(error_analyis)
        f.write(error_analyis)

    return results
