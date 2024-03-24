# adapted from egg.zoo.signal_game
# https://github.com/facebookresearch/EGG/tree/main/egg/zoo/signal_game

import egg.core as core
from experiment.models.agents import InformedSender, Receiver
from options import Options
import torch.nn.functional as F
from analysis.callbacks import DisentAtEnd, ResultsCollector, TopographicSimilarityAtEnd


def get_game(options: Options):
    def loss_nll(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
        nll = F.nll_loss(receiver_output, labels, reduction="none")
        acc = (labels == receiver_output.argmax(dim=1)).float()
        return nll, {"acc": acc}

    sender = InformedSender(options)
    receiver = Receiver(options)

    sender = core.RnnSenderGS(sender, options.vocab_size, options.embedding_size, options.hidden_size, max_len=options.max_len, temperature=1.0, cell=options.sender_cell, trainable_temperature=True)
    receiver = core.RnnReceiverGS(receiver, options.vocab_size, options.embedding_size, options.hidden_size, cell=options.sender_cell)
    game = core.SenderReceiverRnnGS(sender, receiver, loss_nll, length_cost=options.length_cost)
    return game


def perform_training(options: Options, train_loader, valid_loader, game):
    results = []
    callbacks = [ResultsCollector(results, options)]
    if options.enable_analysis:
        callbacks.extend([DisentAtEnd(options), TopographicSimilarityAtEnd(options)])

    trainer = core.Trainer(
        game=game,
        optimizer=core.build_optimizer(game.parameters()),
        train_data=train_loader,
        validation_data=valid_loader,
        callbacks=callbacks,
        device=options.device,
    )

    trainer.train(n_epochs=options.n_epochs)
    core.close()
    return '\n'.join(results), trainer
