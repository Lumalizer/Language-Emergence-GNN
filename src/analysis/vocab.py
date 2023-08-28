import os
import plotly.express as px
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from analysis.message_embeddings import MessageEmbeddings
import torch


def analyze_vocab(title, filename="2023_19_07_13_54_38graphvsimage/vocab2023_19_07_14_46_23___graph_maxlen_4_cellgru_game10_vocab5_hidden60_unseen1_epochs300", specific_target=''):
    path = os.path.join(os.getcwd(), "results/", filename + ".json")
    df = pd.read_json(path)

    messages = df['message'].tolist()
    embedder = MessageEmbeddings(len(messages[0]), 2)
    messages = torch.stack([torch.tensor(message) for message in messages]).type(dtype=torch.float32)
    embedded_messages = embedder.forward(messages)

    distance_matrix = pairwise_distances(embedded_messages, metric='cosine')
    model = TSNE(n_components=2, metric='precomputed', init='random', random_state=42)
    transformed = model.fit_transform(distance_matrix)

    fig = px.scatter(transformed, x=0, y=1, color=df['message'].apply(str).tolist(), hover_name=df['target'].tolist())
    fig.update_layout(title=title)
    fig.show()


def store_vocab_info(options, target_labels, messages, accuracies, target_folder):
    df = pd.DataFrame({'target': target_labels, 'message': messages, 'accuracy': accuracies})

    with open(f'{target_folder}/{"vocab" + str(options)}.json', 'w') as f:
        df.to_json(f)

if __name__ == '__main__':
    analyze_vocab("Vocab distribution")