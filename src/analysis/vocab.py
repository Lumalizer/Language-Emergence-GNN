import os
import numpy as np
import pandas as pd
import nltk
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import plotly.graph_objs as go
import plotly.express as px


def store_vocab_info(options, target_labels, distractor_labels, messages, accuracies, target_folder):
    df = pd.DataFrame({'target': target_labels, 'distractors': distractor_labels, 'message': messages, 'accuracy': accuracies})
    with open(f'{target_folder}/{"vocab" + str(options)}.json', 'w') as f:
        df.to_json(f)


def read_vocab(filename):
    path = os.path.join(os.getcwd(), "results\\", filename + ".json")
    return pd.read_json(path)


def get_tokenizer(n):
    def tokenize_sentences(sentences):
        def get_sentence_ngrams(sentence):
            sentence = list(nltk.ngrams(sentence, n, pad_left=True))
            return [str(s) for s in sentence]
        return [get_sentence_ngrams(sentence) for sentence in sentences]
    return tokenize_sentences


def get_word_frequencies(sentences, show_plot=True):
    words = {}
    for sentence in sentences:
        for word in sentence:
            if word not in words:
                words[word] = 0
            words[word] += 1

    if show_plot:
        fig = px.bar(x=list(words.keys()), y=list(words.values()), title="Word Frequencies", labels={'x': 'Word', 'y': 'Frequency'}, color=list(words.keys()))
        fig.show()

    return words


def visualize_word_embeddings(model: Word2Vec):
    words = list(model.wv.index_to_key)
    word_vectors = np.array([model.wv[word] for word in words])
    pca_result = PCA(n_components=2).fit_transform(word_vectors)
    trace = go.Scatter(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        mode='markers',
        text=words,
        marker=dict(
            size=15,
            color=pca_result[:, 1],
            colorscale='Viridis',
            opacity=0.8
        )
    )
    layout = go.Layout(
        title="Word Embeddings in 2D",
        xaxis=dict(title='Principal Component 1'),
        yaxis=dict(title='Principal Component 2')
    )
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()


def average_similarity(model: Word2Vec):
    vocab = list(model.wv.index_to_key)
    similarities = []
    words = []

    for word in vocab:
        similarity = 0
        for other_word in vocab:
            if word != other_word:
                similarity += model.wv.similarity(word, other_word)
        similarities.append(similarity / (len(vocab) - 1))
        words.append(word)
    fig = px.bar(x=words, y=similarities, title="Average similarity of each word in the vocabulary", labels={'x': 'Word', 'y': 'Average similarity'}, color=similarities)
    fig.show()


def analyze_relative_word_frequencies(df: pd.DataFrame, tokenizer, filter_function):
    # compares the relative word frequencies of the filtered dataset to the original dataset
    # example filter functions: lambda df: df['target'].apply(lambda x: 'bunny' in x) - takes all messages that contain the word 'bunny'
    #                           lambda df: df['accuracy'].apply(lambda x: x != 1.0) - takes all messages that were not guessed correctly

    filtered_df: pd.DataFrame = df[filter_function(df)]

    tokenized_sentences = tokenizer(df['message'].tolist())
    new_tokenized_sentences = tokenizer(filtered_df['message'].tolist())

    word_frequencies = get_word_frequencies(tokenized_sentences, show_plot=False)
    new_frequencies = get_word_frequencies(new_tokenized_sentences, show_plot=False)

    for word in new_frequencies:
        new_frequencies[word] = (new_frequencies[word] / len(filtered_df)) / (word_frequencies[word] / len(df))

    # remove words with frequencies less than 100
    for word in list(new_frequencies.keys()):
        if word_frequencies[word] < 100:
            del new_frequencies[word]

    fig = px.bar(x=list(new_frequencies.keys()), y=list(new_frequencies.values()), title=f"Word Frequencies ({len(filtered_df)} elements)", labels={'x': 'Word', 'y': 'Frequency'}, color=list(new_frequencies.keys()))
    fig.show()


if __name__ == '__main__':
    df = read_vocab("important\\vocab2023_17_09_23_09_05___graph_maxlen_4_cellgru_game5_vocab7_hidden120_unseen1_epochs40")
    tokenizer = get_tokenizer(n=1)
    tokenized_sentences = tokenizer(df['message'].tolist())
    model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, sg=1, epochs=20)

    word_frequencies = get_word_frequencies(tokenized_sentences)
    visualize_word_embeddings(model)
    average_similarity(model)
    analyze_relative_word_frequencies(df, tokenizer, lambda df: df['target'].apply(lambda x: 'bunny' in x))
    analyze_relative_word_frequencies(df, tokenizer, lambda df: df['accuracy'].apply(lambda x: x != 1.0))
