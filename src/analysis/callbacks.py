from egg.core.language_analysis import TopographicSimilarity, Disent
import pickle
import torch
import json
from egg.core import Interaction
from options import Options
import tqdm
import json
from egg.core.callbacks import ConsoleLogger
from options import Options
import wandb
import warnings
from scipy import stats

warnings.filterwarnings('ignore', category=stats.ConstantInputWarning)

class ResultsCollector(ConsoleLogger):
    def __init__(self, results: list, options: Options):
        super().__init__(True, True)
        self.results = results
        self.options = options
        if options.print_progress:
            self.progress_bar = tqdm.tqdm(total=options.n_epochs)

    # adapted from egg.core.callbacks.ConsoleLogger
    def aggregate_print(self, loss: float, logs, mode: str, epoch: int):
        dump = dict(loss=loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)
        dump.update(dict(mode=mode, epoch=epoch))
            
        logged = {f"{dump['mode']}/{k}": v for k, v in sorted(dump.items()) if k not in ["mode", "epoch"]}
        logged['epoch'] = epoch
        wandb.log(logged)

        results = json.dumps(dump)
        self.results.append(results)

        if self.options.print_progress:
            if mode == "train":
                self.progress_bar.update(1)
            else:
                mode = "test"

            output_message = ", ".join(sorted([f"{k}={round(v, 5) if isinstance(v, float) else v}" for k, v in dump.items() if k != "mode"]))
            output_message = f"mode={mode}: " + output_message
            self.progress_bar.set_description(output_message, refresh=True)

def run_or_skip_metrics(epoch, max_epoch):
    return epoch < 5 or not epoch % 2 and epoch < 20 or not epoch % 40 or epoch == max_epoch

class TopographicSimilarityAtEnd(TopographicSimilarity):
    def __init__(self, options: Options):
        super().__init__('hamming','edit', is_gumbel=True, compute_topsim_train_set=True)
        self.options = options

    def on_epoch_end(self, loss: float, logs, epoch: int):
        if run_or_skip_metrics(epoch, self.options.n_epochs):
            super().on_epoch_end(loss, logs, epoch)

    def on_validation_end(self, loss: float, logs, epoch: int):
        if run_or_skip_metrics(epoch, self.options.n_epochs):
            super().on_validation_end(loss, logs, epoch)
    
    def print_message(self, logs: Interaction, mode: str, epoch: int) -> None:
        messages = logs.message.argmax(dim=-1) if self.is_gumbel else logs.message
        messages = [msg.tolist() for msg in messages]

        sender_input = torch.flatten(logs.aux_input['vectors_sender'], start_dim=1).detach()
        topsim = self.compute_topsim(sender_input, messages, self.sender_input_distance_fn, self.message_distance_fn)
        output = json.dumps(dict(topsim=topsim, mode=mode, epoch=epoch))
        # print(output, flush=True)

        logged = {"epoch": epoch, f"{mode}/topsim": topsim}
        wandb.log(logged)

        with open(self.options._target_folder + "/experiments/topsim_" + str(self.options) + ".json", "a") as f:
            f.write(output + "\n")

class DisentAtEnd(Disent):
    def __init__(self, options: Options):
        super().__init__(is_gumbel=True, vocab_size=options.vocab_size, 
                         compute_bosdis=True, compute_posdis=True, print_train=True)
        self.options = options

    def on_epoch_end(self, loss: float, logs, epoch: int):
        if run_or_skip_metrics(epoch, self.options.n_epochs):
            super().on_epoch_end(loss, logs, epoch)

    def on_validation_end(self, loss: float, logs, epoch: int):
        if run_or_skip_metrics(epoch, self.options.n_epochs):
            super().on_validation_end(loss, logs, epoch)

    def print_message(self, logs: Interaction, tag: str, epoch: int):
        message = logs.message.argmax(dim=-1) if self.is_gumbel else logs.message
        sender_input = torch.flatten(logs.aux_input['vectors_sender'], start_dim=1)
    
        posdis = self.posdis(sender_input, message) if self.compute_posdis else None
        bosdis = self.bosdis(sender_input, message, self.vocab_size) if self.compute_bosdis else None
        output = json.dumps(dict(posdis=posdis, bosdis=bosdis, mode=tag, epoch=epoch))
        # print(output, flush=True)

        logged = {"epoch": epoch, f"{tag}/posdis": posdis, f"{tag}/bosdis": bosdis}
        wandb.log(logged)

        with open(self.options._target_folder + "/experiments/dissent_" + str(self.options) + ".json", "a") as f:
            f.write(output + "\n")
