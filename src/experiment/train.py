import egg.core as core
from options import Options
from analysis.timer import timer
from analysis.callbacks import ResultsCollector, DisentAtEnd, TopographicSimilarityAtEnd


@timer
def perform_training(options: Options, train_loader, valid_loader, game):
    results = []

    trainer = core.Trainer(
        game=game,
        optimizer=core.build_optimizer(game.parameters()),
        train_data=train_loader,
        validation_data=valid_loader,
        callbacks=[ResultsCollector(results, options), DisentAtEnd(options), TopographicSimilarityAtEnd(options)],
        device=options.device,
    )

    trainer.train(n_epochs=options.n_epochs)
    core.close()
    return '\n'.join(results), trainer
