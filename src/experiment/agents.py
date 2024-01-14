# inspired by egg.zoo.signal_game.archs

import torch
import torch.nn as nn
import torch.nn.functional as F
from options import ExperimentOptions
from data.graph.graph_embeddings import GraphEmbeddings
from data.image.image_embeddings import ImageEmbeddings
from data.graph.graph_builder import GraphBuilder
from data.image.image_loader import ImageLoader

graph_embedder = GraphEmbeddings(30)
image_embedder = ImageEmbeddings(30)

class InformedSender(nn.Module):
    def __init__(self, options: ExperimentOptions, label_codes: dict[int, str]):
        super(InformedSender, self).__init__()
        self.options = options
        self.label_decoder = label_codes
        self.builder = GraphBuilder(embedding_size=options.embedding_size) if options.experiment == 'graph' else ImageLoader()
        self.embedder = graph_embedder if self.options.experiment == 'graph' else image_embedder

        self.conv1 = nn.Conv2d(1, options.hidden_size, kernel_size=(options.game_size, 1), stride=(options.game_size, 1), bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(options.hidden_size, 1), stride=(options.hidden_size, 1), bias=False)
        self.lin1 = nn.Linear(options.embedding_size, options.hidden_size, bias=False)

        self.rl1 = torch.nn.LeakyReLU()
        self.rl2 = torch.nn.LeakyReLU()

    def forward(self, x, _aux_input=None):
        if not self.options.use_prebuilt_embeddings:
            x = self.builder.get_batched_data([self.label_decoder[i] for sublist in x.tolist() for i in sublist])
            x = x.to(self.options.device)
            x = self.embedder.forward(x, detach=self.options.use_prebuilt_embeddings)
            x = x.view(self.options.batch_size, self.options.game_size, -1)
            
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
    def __init__(self, options: ExperimentOptions, label_codes: dict[int, str]):
        super(Receiver, self).__init__()
        self.options = options
        self.label_decoder = label_codes
        self.builder = GraphBuilder(embedding_size=options.embedding_size) if options.experiment == 'graph' else ImageLoader()
        self.embedder = graph_embedder if self.options.experiment == 'graph' else image_embedder
        self.lin1 = nn.Linear(options.hidden_size, options.embedding_size)

    def forward(self, signal, x, _aux_input=None):
        if not self.options.use_prebuilt_embeddings:
            x = self.builder.get_batched_data([self.label_decoder[i] for sublist in x.tolist() for i in sublist])
            x = x.to(self.options.device)
            x = self.embedder.forward(x, detach=self.options.use_prebuilt_embeddings)
            x = x.view(self.options.batch_size, self.options.game_size, -1)

        h_s = self.lin1(signal)                 # embed the signal
        h_s = h_s.unsqueeze(dim=1)              # batch_size x embedding_size
        h_s = h_s.transpose(1, 2)               # batch_size x 1 x embedding_size
        out = torch.bmm(x, h_s)                 # batch_size x embedding_size x 1
        out = out.squeeze(dim=-1)               # batch_size x game_size x 1
        log_probs = F.log_softmax(out, dim=1)   # batch_size x game_size
        return log_probs