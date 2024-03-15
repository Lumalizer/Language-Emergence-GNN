import egg.core as core
from options import Options
from analysis.timer import timer
from analysis.logger import ResultsCollector
from analysis.language_analysis import DisentAtEnd, TopographicSimilarityAtEnd


@timer
def perform_training(options: Options, train_loader, valid_loader, game):
    results = []

    trainer = core.Trainer(
        game=game,
        optimizer=core.build_optimizer(game.parameters()),
        train_data=train_loader,
        validation_data=valid_loader,
        callbacks=[ResultsCollector(results, options), 
                   DisentAtEnd(options.n_epochs), 
                   TopographicSimilarityAtEnd(options.n_epochs)],
        device=options.device,
    )

    trainer.train(n_epochs=options.n_epochs)
    core.close()
    return '\n'.join(results), trainer
