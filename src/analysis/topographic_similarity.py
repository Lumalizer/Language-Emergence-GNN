from egg.core.language_analysis import TopographicSimilarity
import pickle

def topsim_single(interaction):
    if interaction.endswith('.pkl'):
        with open(f'results/{interaction}', 'rb') as f:
            interaction = pickle.load(f)

    topsim = TopographicSimilarity('hamming', 'edit', is_gumbel=True)
    topsim.print_message(interaction, 'gumbel', 0)

def topsim_double_swap(interaction1, interaction2):
    if interaction1.endswith('.pkl'):
        with open(f'results/{interaction1}', 'rb') as f:
            interaction1 = pickle.load(f)

    if interaction2.endswith('.pkl'):
        with open(f'results/{interaction2}', 'rb') as f:
            interaction2 = pickle.load(f)

    topsim = TopographicSimilarity('hamming', 'edit', is_gumbel=True)
    interaction1.sender_input = interaction2.sender_input
    topsim.print_message(interaction1, 'gumbel', 0)

class TopographicSimilarityAtEnd(TopographicSimilarity):
    def __init__(self, n_epochs):
        super().__init__('hamming','edit', is_gumbel=True)
        self.n_epochs = n_epochs

    def on_epoch_end(self, loss: float, logs, epoch: int):
        pass

    def on_validation_end(self, loss: float, logs, epoch: int):
        return
        if epoch == self.n_epochs:
            super().on_validation_end(loss, logs, epoch)


if __name__ == '__main__':
    # topsim_single('2024_16_01_17_53_28graphvsimage/experiments/interaction_2024_16_01_17_53_28___graph_maxlen_4_vocab7_game5.pkl')
    topsim_double_swap('2024_16_01_17_53_28graphvsimage/experiments/interaction_2024_16_01_17_53_28___graph_maxlen_4_vocab7_game5.pkl', 
                       '2024_16_01_17_53_28graphvsimage/experiments/interaction_2024_16_01_17_53_28___image_maxlen_4_vocab7_game5.pkl')