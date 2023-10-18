import os
import numpy as np
import pandas as pd
import nltk
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import plotly.graph_objs as go
import plotly.express as px


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


def has_challenge(target, distractors):
    target = target.split('_')
    distractors_ = [d.split('_') for d in distractors]
    same_positions = [sum(t == d == '0' for t, d in zip(target, distractor)) > 1 for distractor in distractors_]

    target = [t for t in target if t != '0']
    same_shapes = [sorted(target) == sorted([d for d in distractor.split('_') if d != '0']) for distractor in distractors]
    return sum(same_positions) > 0, sum(same_shapes) > 0


def vocab_error_analysis(df: pd.DataFrame) -> str:
    associations = df.groupby('target')['message'].apply(lambda x: set(str(msg) for msg in x)).to_dict()
    # columns = ['target', 'distractors', 'message', 'accuracy']
    df['has_same_positions'], df['has_same_shapes'] = zip(*df.apply(lambda row: has_challenge(row['target'], row['distractors']), axis=1))
    errors = df[df['accuracy'] != 1.0]

    result = ""
    result += f"{df.head(10)}\n\n"
    result += f"Total games: {len(df)} (accuracy: {df['accuracy'].mean()*100:.2f}%)\n"
    result += f"Mean messages per target: {sum(len(v) for v in associations.values()) / len(associations)} (unique targets: {len(df['target'].unique())})\n"
    result += f"Total unique messages: {len(df['message'].apply(str).unique())}\n"
    result += f"Total unique messages in errors: {len(errors['message'].apply(str).unique())}\n"

    has_same_position = df[df['has_same_positions'] & ~(df['has_same_shapes'])]
    has_same_shape = df[df['has_same_shapes'] & ~(df['has_same_positions'])]
    has_same_shapes_and_position = df[(df['has_same_positions']) & (df['has_same_shapes'])]
    has_everything_different = df[~(df['has_same_positions']) & ~(df['has_same_shapes'])]

    result += f"Same positions only: {len(has_same_position)} ({len(has_same_position) / len (df)*100:.2f}% of all games) ({has_same_position['accuracy'].mean()*100:.2f}% correct)\n"
    result += f"Same shapes only: {len(has_same_shape)} ({len(has_same_shape) / len(df)*100:.2f}% of all games) ({has_same_shape['accuracy'].mean()*100:.2f}% correct)\n"
    result += f"Same positions and shapes: {len(has_same_shapes_and_position)} ({len(has_same_shapes_and_position) / len(df)*100:.2f}% of all games) ({has_same_shapes_and_position['accuracy'].mean()*100:.2f}% correct)\n"
    result += f"Everything different: {len(has_everything_different)} ({len(has_everything_different) / len(df)*100:.2f}% of all games) ({has_everything_different['accuracy'].mean()*100:.2f}% correct)\n"

    return result


def show_error_targets(df: pd.DataFrame):
    errors = df[df['accuracy'] != 1.0]
    error_targets = errors['target'].tolist()
    error_targets = [t.split('_') for t in error_targets]
    error_targets = [item for sublist in error_targets for item in sublist if item != '0']
    fig = px.histogram(x=error_targets, title="Error Analysis", labels={'x': 'Word', 'y': 'Frequency'}, color=error_targets)
    fig.show()


if __name__ == '__main__':
    df = read_vocab("important\\vocab2023_17_10_16_45_15___graph_maxlen_3_vocab60_game5")
    # tokenizer = get_tokenizer(n=1)
    # tokenized_sentences = tokenizer(df['message'].tolist())
    # model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, sg=1, epochs=20)
    # word_frequencies = get_word_frequencies(tokenized_sentences)
    # visualize_word_embeddings(model)
    # average_similarity(model)
    # analyze_relative_word_frequencies(df, tokenizer, lambda df: df['target'].apply(lambda x: 'bunny' in x))
    # analyze_relative_word_frequencies(df, tokenizer, lambda df: df['accuracy'].apply(lambda x: x != 1.0))
    print(vocab_error_analysis(df))
