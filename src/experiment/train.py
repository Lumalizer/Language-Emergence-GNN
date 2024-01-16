import egg.core as core
from options import ExperimentOptions
from analysis.timer import timer
from analysis.logger import ResultsCollector
from analysis.topographic_similarity import TopographicSimilarityAtEnd


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
