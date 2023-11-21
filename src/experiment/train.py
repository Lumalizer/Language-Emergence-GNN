import egg.core as core
from options import ExperimentOptions
from analysis.timer import timer
from analysis.logger import ResultsCollector
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


@timer
def perform_training(options: ExperimentOptions, train_loader, valid_loader, game):
    results = []

    trainer = core.Trainer(
        game=game,
        optimizer=core.build_optimizer(game.parameters()),
        train_data=train_loader,
        validation_data=valid_loader,
        callbacks=[ResultsCollector(results, options), TopographicSimilarityAtEnd(options.n_epochs)]
    )

    trainer.train(n_epochs=options.n_epochs)
    core.close()
    return '\n'.join(results), trainer
