import os
import logging
from random import sample
from itertools import combinations
from options import ExperimentOptions
from data.datastring_builder import DatastringBuilder


def split_data_labels(options: ExperimentOptions):
    labels = DatastringBuilder().datastrings
    label_codes = {i: l for i, l in enumerate(labels)}

    if options.n_unseen_shapes == 0:
        train_labels = labels[len(labels)//4:]
        valid_labels = labels[:len(labels)//4]
    else:
        shapes = [s.replace('.png', '') for s in os.listdir('../assets/shapes') if s.endswith('.png')]
        separated = sample(shapes, options.n_separated_shapes)

        train_labels = [l for l in labels if not any([s in l for s in separated])]
        other_labels = [l for l in labels if l not in train_labels]

        if options.n_unseen_shapes == 1:
            valid_labels = set([l for l in other_labels if not any([all([s in l for s in c]) for c in combinations(separated, 2)])])
            extra_labels = set([l for l in other_labels if l not in valid_labels])
            logging.warning(f"Removed {len(extra_labels)} labels from valid set because they contained more than one of {separated}")
        elif options.n_unseen_shapes == 2:
            valid_labels = list(set([l for l in other_labels if any([all([s in l for s in c]) for c in combinations(separated, 2)])]))
            logging.warning(f"{len(valid_labels)} elements in valid. Removed {len(other_labels) - len(valid_labels)} labels from valid set because they contained less than one of {separated}")

    logging.warning(f"Train size {len(train_labels)*100 // len(labels)}% ({len(train_labels)}), valid size {len(valid_labels)*100 // len(labels)}% ({len(valid_labels)})")

    assert len(set(train_labels).intersection(set(valid_labels))) == 0
    return train_labels, valid_labels, label_codes
