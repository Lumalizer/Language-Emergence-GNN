import egg.core as core
from options import ExperimentOptions
from analysis.timer import timer
from analysis.logger import ResultsCollector


@timer
def perform_training(options: ExperimentOptions, train_loader, valid_loader, game):
    results = []

    trainer = core.Trainer(
        game=game,
        optimizer=core.build_optimizer(game.parameters()),
        train_data=train_loader,
        validation_data=valid_loader,
        callbacks=[ResultsCollector(results, print_to_console=options.print_to_console)],
    )

    trainer.train(n_epochs=options.n_epochs)
    core.close()
    return '\n'.join(results)
