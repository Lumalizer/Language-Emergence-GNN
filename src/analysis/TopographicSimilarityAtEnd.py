from egg.core.language_analysis import TopographicSimilarity


class TopographicSimilarityAtEnd(TopographicSimilarity):
    def __init__(self, n_epochs):
        super().__init__('hamming','edit', is_gumbel=True)
        self.n_epochs = n_epochs

    def on_epoch_end(self, loss: float, logs, epoch: int):
        pass

    def on_validation_end(self, loss: float, logs, epoch: int):
        if epoch == self.n_epochs:
            super().on_validation_end(loss, logs, epoch)