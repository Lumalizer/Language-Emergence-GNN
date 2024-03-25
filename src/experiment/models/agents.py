# inspired by egg.zoo.signal_game.archs
import torch
import torch.nn as nn
import torch.nn.functional as F
from options import Options
from experiment.models.graph_embeddings import GraphEmbeddings
from experiment.models.image_embeddings import ImageEmbeddings

class InformedSender(nn.Module):
    def __init__(self, options: Options):
        super(InformedSender, self).__init__()
        self.options = options
        self.view_size = options.game_size if not options.sender_target_only else 1
        self.embedder = GraphEmbeddings(options.embedding_size) if options.experiment == 'graph' else ImageEmbeddings(options.embedding_size)

        self.conv1 = nn.Conv2d(1, options.hidden_size, kernel_size=(self.view_size, 1), stride=(self.view_size, 1), bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(options.hidden_size, 1), stride=(options.hidden_size, 1), bias=False)
        self.lin1 = nn.Linear(options.embedding_size, options.hidden_size, bias=False)

        self.rl1 = torch.nn.LeakyReLU()
        self.rl2 = torch.nn.LeakyReLU()

    def forward(self, x, _aux_input):
        x = self.embedder(_aux_input['data_sender'])
        x = x.view(self.options.batch_size, self.view_size, -1)
        # _aux_input['vectors_sender'] = x

        emb = x.unsqueeze(dim=1)                # batch_size x 1 x game_size x embedding_size
        h = self.conv1(emb)                     # batch_size x hidden_size x 1 x embedding_size
        h = self.rl1(h)
        h = h.transpose(1, 2)                   # batch_size, 1, hidden_size, embedding_size
        h = self.conv2(h)                       # batch_size, 1, 1, embedding_size
        h = self.rl2(h)
        h = h.squeeze()                         # batch_size x embedding_size
        h = self.lin1(h)                        # batch_size x hidden_size
        h = h.mul(1.0 / self.options.tau_s)
        return h

class Receiver(nn.Module):
    def __init__(self, options: Options):
        super(Receiver, self).__init__()
        self.options = options
        self.embedder = GraphEmbeddings(options.embedding_size) if options.experiment == 'graph' else ImageEmbeddings(options.embedding_size)
        self.lin1 = nn.Linear(options.hidden_size, options.embedding_size)

    def forward(self, signal, x, _aux_input):
        x = self.embedder(_aux_input['data_receiver'])
        x = x.view(self.options.batch_size, self.options.game_size, -1)
        # _aux_input['vectors_receiver'] = x

        h_s = self.lin1(signal)                 # embed the signal
        h_s = h_s.unsqueeze(dim=1)              # batch_size x embedding_size
        h_s = h_s.transpose(1, 2)               # batch_size x 1 x embedding_size
        out = torch.bmm(x, h_s)                 # batch_size x embedding_size x 1
        out = out.squeeze(dim=-1)               # batch_size x game_size x 1
        log_probs = F.log_softmax(out, dim=1)   # batch_size x game_size
        return log_probs