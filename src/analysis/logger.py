import json
from egg.core.callbacks import ConsoleLogger


class ResultsCollector(ConsoleLogger):
    def __init__(self, results: list, print_to_console: bool, print_train_loss=True, as_json=True):
        super().__init__(as_json=as_json, print_train_loss=print_train_loss)
        self.results = results
        self.print_to_console = print_to_console

    # adapted from egg.core.callbacks.ConsoleLogger
    def aggregate_print(self, loss: float, logs, mode: str, epoch: int):
        dump = dict(loss=loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)
        dump.update(dict(mode=mode, epoch=epoch))

        results = json.dumps(dump)
        self.results.append(results)

        if self.print_to_console:
            output_message = ", ".join(sorted([f"{k}={v}" for k, v in dump.items()]))
            output_message = f"{mode}: epoch {epoch}, loss {loss}, " + output_message
            print(output_message)
