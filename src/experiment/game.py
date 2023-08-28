# adapted from egg.zoo.signal_game
# https://github.com/facebookresearch/EGG/tree/main/egg/zoo/signal_game

import egg.core as core
from experiment.agents import InformedSender, Receiver
from options import ExperimentOptions
import torch.nn.functional as F


def get_game(options: ExperimentOptions):
    def loss_nll(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
        nll = F.nll_loss(receiver_output, labels, reduction="none")
        acc = (labels == receiver_output.argmax(dim=1)).float()
        return nll, {"acc": acc}

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
        options.hidden_size)

    sender = core.RnnSenderReinforce(sender, options.vocab_size, options.embedding_size,
                                     options.hidden_size, max_len=options.max_len, cell=options.sender_cell)
    receiver = core.RnnReceiverDeterministic(receiver, options.vocab_size, options.embedding_size,
                                             options.hidden_size)
    game = core.SenderReceiverRnnReinforce(
        sender,
        receiver,
        loss_nll,
        sender_entropy_coeff=0.01,
        receiver_entropy_coeff=0.01,
    )
    return game
