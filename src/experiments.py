from options import ExperimentOptions

experiments = [
    ExperimentOptions(experiment='both', game_size=2, max_len=1, vocab_size=5, sender_cell='gru'),
    ExperimentOptions(experiment='both', game_size=5, max_len=1, vocab_size=5, sender_cell='gru'),
    ExperimentOptions(experiment='both', game_size=10, max_len=1, vocab_size=5, sender_cell='gru'),
    ExperimentOptions(experiment='both', game_size=25, max_len=1, vocab_size=5, sender_cell='gru'),
    ExperimentOptions(experiment='both', game_size=50, max_len=1, vocab_size=5, sender_cell='gru'),

    ExperimentOptions(experiment='both', game_size=2, max_len=4, vocab_size=5, sender_cell='gru'),
    ExperimentOptions(experiment='both', game_size=5, max_len=4, vocab_size=5, sender_cell='gru'),
    ExperimentOptions(experiment='both', game_size=10, max_len=4, vocab_size=5, sender_cell='gru'),
    ExperimentOptions(experiment='both', game_size=25, max_len=4, vocab_size=5, sender_cell='gru'),
    ExperimentOptions(experiment='both', game_size=50, max_len=4, vocab_size=5, sender_cell='gru'),

    ExperimentOptions(experiment='both', game_size=2, max_len=8, vocab_size=5, sender_cell='gru'),
    ExperimentOptions(experiment='both', game_size=5, max_len=8, vocab_size=5, sender_cell='gru'),
    ExperimentOptions(experiment='both', game_size=10, max_len=8, vocab_size=5, sender_cell='gru'),
    ExperimentOptions(experiment='both', game_size=25, max_len=8, vocab_size=5, sender_cell='gru'),
    ExperimentOptions(experiment='both', game_size=50, max_len=8, vocab_size=5, sender_cell='gru')
]
