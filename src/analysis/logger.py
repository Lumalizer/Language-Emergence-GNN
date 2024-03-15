import tqdm
import json
from egg.core.callbacks import ConsoleLogger
from options import Options


class ResultsCollector(ConsoleLogger):
    def __init__(self, results: list, options: Options):
        super().__init__(True, True)
        self.results = results
        self.options = options
        self.progress_bar = tqdm.tqdm(total=options.n_epochs)

    # adapted from egg.core.callbacks.ConsoleLogger
    def aggregate_print(self, loss: float, logs, mode: str, epoch: int):
        dump = dict(loss=loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)
        dump.update(dict(mode=mode, epoch=epoch))

        results = json.dumps(dump)
        self.results.append(results)

        if self.options.print_to_console:
            if mode == "train":
                self.progress_bar.update(1)
            else:
                mode = " test"

            output_message = ", ".join(sorted([f"{k}={round(v, 5) if isinstance(v, float) else v}" for k, v in dump.items() if k != "mode"]))
            output_message = f"mode={mode}: " + output_message
            self.progress_bar.set_description(output_message, refresh=True)
