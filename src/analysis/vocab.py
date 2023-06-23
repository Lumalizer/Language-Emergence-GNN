import os
import plotly.express as px
import json
import pandas as pd


def analyze_vocab(title, filename, specific_target=''):
    path = os.path.join(os.getcwd(), "results/", filename + ".json")
    data: dict[str, list[str, list[int]]] = json.load(open(path))

    data = [v for i, (k, v) in enumerate(data.items()) if sum(v[1]) and i > len(data)//1.5]

    # print(sorted(data))
    print(len(data))
    print(data[0])

    targets = [p[0] for p in data]

    if not specific_target:
        words = [str(p[1]) for p in data]
    else:
        words = [str(p[1]) for p in data if p[0] == specific_target]
        title = title + f" for '{specific_target}'"

    fig = px.histogram(words, title=title, labels={'value': 'sentence', 'count': 'count'})
    fig.update_layout(showlegend=False)
    fig.update_layout(font=dict(size=32))
    fig.update_xaxes(categoryorder="total descending")
    fig.show()


def store_vocab_info(options, target_labels, messages, target_folder):
    m = []
    for batch in messages:
        m.extend(batch)

    d = {i: (str(target_labels[i]), m[i].tolist()) for i in range(len(target_labels))}

    with open(f'{target_folder}/{"vocab" + str(options)}.json', 'w') as f:
        json.dump(d, f)
