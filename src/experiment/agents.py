# inspired by egg.zoo.signal_game.archs

import torch
import torch.nn as nn
import torch.nn.functional as F


class InformedSender(nn.Module):
    def __init__(self, game_size, feat_size, embedding_size, hidden_size, vocab_size, temp):
        super(InformedSender, self).__init__()
        self.game_size = game_size
        self.temp = temp

        self.lin1 = nn.Linear(feat_size, embedding_size, bias=False)
        self.conv2 = nn.Conv2d(1, hidden_size, kernel_size=(game_size, 1), stride=(game_size, 1), bias=False)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=(hidden_size, 1), stride=(hidden_size, 1), bias=False)
        self.lin4 = nn.Linear(embedding_size, hidden_size, bias=False)

    def forward(self, x, _aux_input=None):
        emb = self.return_embeddings(x)         # batch_size x 1 x game_size x embedding_size
        h = self.conv2(emb)                     # batch_size x hidden_size x 1 x embedding_size
        h = torch.nn.LeakyReLU()(h)
        h = h.transpose(1, 2)                   # batch_size, 1, hidden_size, embedding_size
        h = self.conv3(h)                       # batch_size, 1, 1, embedding_size
        h = torch.nn.LeakyReLU()(h)
        h = h.squeeze()                         # batch_size x embedding_size
        h = self.lin4(h)                        # batch_size x hidden_size
        h = h.mul(1.0 / self.temp)
        return h

    def return_embeddings(self, x):
        embs = []
        x = x.transpose(0, 1)
        for i in range(self.game_size):
            h_i = self.lin1(x[i])               # batch_size x embedding_size
            h_i = h_i.unsqueeze(dim=1)          # batch_size x 1 x embedding_size
            h_i = h_i.unsqueeze(dim=1)          # batch_size x 1 x 1 x embedding_size
            embs.append(h_i)
        return torch.cat(embs, dim=2)           # batch_size x 1 x game_size x embedding_size


class Receiver(nn.Module):
    def __init__(self, game_size, feat_size, embedding_size, hidden_size):
        super(Receiver, self).__init__()
        self.game_size = game_size

        self.lin1 = nn.Linear(feat_size, embedding_size, bias=False)
        self.lin2 = nn.Linear(hidden_size, embedding_size)

    def forward(self, signal, x, _aux_input=None):
        emb = self.return_embeddings(x)
        h_s = self.lin2(signal)                 # embed the signal
        h_s = h_s.unsqueeze(dim=1)              # batch_size x embedding_size
        h_s = h_s.transpose(1, 2)               # batch_size x 1 x embedding_size
        out = torch.bmm(emb, h_s)               # batch_size x embedding_size x 1
        out = out.squeeze(dim=-1)               # batch_size x game_size x 1
        log_probs = F.log_softmax(out, dim=1)   # batch_size x game_size
        return log_probs

    def return_embeddings(self, x):
        embs = []
        x = x.transpose(0, 1)
        for i in range(self.game_size):
            h_i = self.lin1(x[i])               # batch_size x embedding_size
            h_i = h_i.unsqueeze(dim=1)          # batch_size x 1 x embedding_size
            embs.append(h_i)
        return torch.cat(embs, dim=1)           # batch_size x 1 x game_size x embedding_size
