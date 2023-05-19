# adapted from egg.zoo.signal_game
# https://github.com/facebookresearch/EGG/tree/main/egg/zoo/signal_game

import egg.core as core
from egg.zoo.signal_game.archs import InformedSender, Receiver
from egg.zoo.signal_game.train import loss
from options import ExperimentOptions


def get_game(options: ExperimentOptions):
    sender = InformedSender(
        options.game_size,
        options.feat_size,
        options.embedding_size,
        options.hidden_size,
        options.vocab_size,
        temp=options.tau_s)
    receiver = Receiver(
        options.game_size,
        options.feat_size,
        options.embedding_size,
        options.vocab_size,
        reinforce=True)

    sender = core.ReinforceWrapper(sender)
    receiver = core.ReinforceWrapper(receiver)
    game = core.SymbolGameReinforce(
        sender,
        receiver,
        loss,
        sender_entropy_coeff=0.01,
        receiver_entropy_coeff=0.01,
    )
    return game
