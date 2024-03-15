from egg.core.language_analysis import TopographicSimilarity, Disent
import pickle
import torch
import json
from egg.core import Interaction
from options import Options

def topsim_single(interaction):
    if interaction.endswith('.pkl'):
        with open(f'results/{interaction}', 'rb') as f:
            interaction = pickle.load(f)

    print(interaction.aux_input)
    interaction.sender_input = interaction.aux_input['vectors_sender']
    interaction.receiver_input = interaction.aux_input['vectors_receiver']
    topsim = TopographicSimilarity('hamming', 'edit', is_gumbel=True)
    topsim.print_message(interaction, 'gumbel', 0)

def topsim_double_swap(interaction1, interaction2):
    if interaction1.endswith('.pkl'):
        with open(f'results/{interaction1}', 'rb') as f:
            interaction1 = pickle.load(f)

    if interaction2.endswith('.pkl'):
        with open(f'results/{interaction2}', 'rb') as f:
            interaction2 = pickle.load(f)

    topsim = TopographicSimilarity('hamming', 'edit', is_gumbel=True)
    interaction1.sender_input = interaction2.sender_input
    topsim.print_message(interaction1, 'gumbel', 0)

class TopographicSimilarityAtEnd(TopographicSimilarity):
    def __init__(self, options: Options):
        super().__init__('hamming','edit', is_gumbel=True)
        self.options = options
        self.n_epochs = options.n_epochs

    def on_epoch_end(self, loss: float, logs, epoch: int):
        pass

    def on_validation_end(self, loss: float, logs, epoch: int):
        if epoch == self.n_epochs:
            super().on_validation_end(loss, logs, epoch)
    
    def print_message(self, logs: Interaction, mode: str, epoch: int) -> None:
        messages = logs.message.argmax(dim=-1) if self.is_gumbel else logs.message
        messages = [msg.tolist() for msg in messages]

        sender_input = torch.flatten(logs.aux_input['vectors_sender'], start_dim=1)

        topsim = self.compute_topsim(sender_input, messages, self.sender_input_distance_fn, self.message_distance_fn)

        output = json.dumps(dict(topsim=topsim, mode=mode, epoch=epoch))
        print(output, flush=True)

        with open(self.options._target_folder + "/experiments/topsim_" + str(self.options) + ".json", "w") as f:
            json.dump(output, f)

class DisentAtEnd(Disent):
    def __init__(self, options: Options):
        super().__init__(is_gumbel=True, vocab_size=options.vocab_size, compute_bosdis=True, compute_posdis=True)
        self.options = options
        self.n_epochs = options.n_epochs

    def on_epoch_end(self, loss: float, logs, epoch: int):
        pass

    def on_validation_end(self, loss: float, logs, epoch: int):
        if epoch == self.n_epochs:
            super().on_validation_end(loss, logs, epoch)

    def print_message(self, logs: Interaction, tag: str, epoch: int):
        message = logs.message.argmax(dim=-1) if self.is_gumbel else logs.message

        sender_input = torch.flatten(logs.aux_input['vectors_sender'], start_dim=1)
        
        posdis = self.posdis(sender_input, message) if self.compute_posdis else None
        bosdis = self.bosdis(sender_input, message, self.vocab_size) if self.compute_bosdis else None
        

        output = json.dumps(dict(posdis=posdis, bosdis=bosdis, mode=tag, epoch=epoch))
        print(output, flush=True)

        with open(self.options._target_folder + "/experiments/dissent_" + str(self.options) + ".json", "w") as f:
            json.dump(output, f)


if __name__ == '__main__':
    topsim_single('2024_02_02_20_10_46graphvsimage/experiments/interaction_2024_02_02_20_10_46___graph_maxlen_4_vocab7_game5.pkl')
    # topsim_double_swap('2024_16_01_17_53_28graphvsimage/experiments/interaction_2024_16_01_17_53_28___graph_maxlen_4_vocab7_game5.pkl', 
    #                    '2024_16_01_17_53_28graphvsimage/experiments/interaction_2024_16_01_17_53_28___image_maxlen_4_vocab7_game5.pkl')