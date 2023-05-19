import os
from datetime import datetime
import plotly.express as px
import pandas as pd


def plot_train_data(df: pd.DataFrame):
    mask = df['mode'] == 'train'
    df = df[mask]
    mask = df['n_unseen_shapes'] == 1
    df = df[mask]
    plot_dataframe(df, 'Train data', mode='train')


def plot_test_data(df: pd.DataFrame, vocab_size: int = 10):
    mask = df['vocab_size'] == vocab_size
    df = df[mask]
    mask = df['mode'] == 'test'
    df = df[mask]
    plot_dataframe(df, f'Test data (vocab {vocab_size})')


def plot_dataframe(df: pd.DataFrame, title: str, show_plot=True, save=True, mode='test'):
    colors = ["#FFA500", "#6495ED"] if mode == 'test' else ["#FF2D00", "#3556BB"]

    figure = px.line(df, x='epoch', y='acc', facet_col='n_unseen_shapes' if mode == 'test' else 'vocab_size',
                  facet_row='game_size', color='experiment', color_discrete_sequence=colors,
                  facet_col_spacing=0.04, facet_row_spacing=0.03)
    figure.update_traces(mode='lines+markers', marker=dict(size=4,
                      line=dict(width=1)), line=dict(width=6))
    figure.update_yaxes(range=[0, 1])
    figure.update_xaxes(range=[0, 50])
    figure.update_layout(
        font=dict(family="Arial", size=26, color='#000000'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    title=title, font=dict(size=32), itemsizing='trace'),
        width=1200, height=1680,
        margin=dict(l=0, r=0, t=0, b=0))

    figure['layout']['xaxis']['title']['text'] = ''
    figure['layout']['xaxis2']['title']['text'] = 'epochs (1024 games/epoch)'
    figure['layout']['xaxis3']['title']['text'] = ''

    show_plot and figure.show()

    if save:
        if not os.path.isdir("results"):
            os.makedirs("results")
        title = title.replace('<br>', '')
        now = datetime.now().strftime("%d_%m_%Y__%H_%M_%S__")
        # orca requires external installation, can use pip install kaleido instead
        figure.write_image(f"results/{now}_{str(title)}.png", engine="orca")
