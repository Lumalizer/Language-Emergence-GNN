# inspired by egg.zoo.signal_game.archs

import torch
import torch.nn as nn
import torch.nn.functional as F


class InformedSender(nn.Module):
    def __init__(self, game_size, embedding_size, hidden_size, vocab_size, temp):
        super(InformedSender, self).__init__()
        self.game_size = game_size
        self.temp = temp

        self.conv1 = nn.Conv2d(1, hidden_size, kernel_size=(game_size, 1), stride=(game_size, 1), bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(hidden_size, 1), stride=(hidden_size, 1), bias=False)
        self.lin1 = nn.Linear(embedding_size, hidden_size, bias=False)

        self.rl1 = torch.nn.LeakyReLU()
        self.rl2 = torch.nn.LeakyReLU()

    def forward(self, x, _aux_input=None):
        emb = x.unsqueeze(dim=1)                # batch_size x 1 x game_size x embedding_size
        h = self.conv1(emb)                     # batch_size x hidden_size x 1 x embedding_size
        h = self.rl1(h)
        h = h.transpose(1, 2)                   # batch_size, 1, hidden_size, embedding_size
        h = self.conv2(h)                       # batch_size, 1, 1, embedding_size
        h = self.rl2(h)
        h = h.squeeze()                         # batch_size x embedding_size
        h = self.lin1(h)                        # batch_size x hidden_size
        h = h.mul(1.0 / self.temp)
        return h

class Receiver(nn.Module):
    def __init__(self, game_size, embedding_size, hidden_size):
        super(Receiver, self).__init__()
        self.game_size = game_size
        self.lin1 = nn.Linear(hidden_size, embedding_size)

    def forward(self, signal, x, _aux_input=None):
        h_s = self.lin1(signal)                 # embed the signal
        h_s = h_s.unsqueeze(dim=1)              # batch_size x embedding_size
        h_s = h_s.transpose(1, 2)               # batch_size x 1 x embedding_size
        out = torch.bmm(x, h_s)                 # batch_size x embedding_size x 1
        out = out.squeeze(dim=-1)               # batch_size x game_size x 1
        log_probs = F.log_softmax(out, dim=1)   # batch_size x game_size
        return log_probs