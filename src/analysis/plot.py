import os
from datetime import datetime
import plotly.express as px
import pandas as pd


def extract_train_data(df: pd.DataFrame):
    mask = df['mode'] == 'train'
    return df[mask]


def extract_test_data(df: pd.DataFrame):
    mask = df['mode'] == 'test'
    return df[mask]


def plot_dataframe(df: pd.DataFrame, title: str, show_plot=True, save=True, mode='both', facet_col="max_len", facet_row="game_size"):
    colors = ["#FFA500", "#6495ED"]

    if mode == 'test':
        df = extract_test_data(df)
    elif mode == 'train':
        colors = ["#FF2D00", "#3556BB"]
        df = extract_train_data(df)

    figure = px.line(df, x='epoch', y='acc', facet_col=facet_col,
                     facet_row=facet_row, color='experiment', color_discrete_sequence=colors,
                     facet_col_spacing=0.01, facet_row_spacing=0.01)

    # Add a horizontal line to each subplot representing the chance level
    for i, facet_row_value in enumerate(reversed(df[facet_row].unique()), start=1):
        for j, facet_col_value in enumerate(df[facet_col].unique(), start=1):
            chance_level = 1 / facet_row_value

            figure.add_shape(type='line',
                            x0=figure.data[0]['x'].min(), x1=figure.data[0]['x'].max(),
                            y0=chance_level, y1=chance_level,
                            line=dict(color='darkgray', width=5, dash='dash'),
                            row=i, col=j)

    # Add a dummy legend item for the chance level
    figure.add_trace(dict(x=[None], y=[None], name='Chance level', line=dict(color='darkgray', width=5), 
                          mode='lines', showlegend=True, legendgroup='legend', hoverinfo='skip'))


    figure.update_layout(
        yaxis=dict(range=[-0.05, 1.05]),
        showlegend=True,
        font=dict(family="Arial", size=34, color='#000000'),
        width=1500, height=2100,
        margin=dict(l=0, r=10, t=0, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    title=title, font=dict(size=32), itemsizing='trace')
    )

    figure.update_traces(mode='lines+markers', marker=dict(size=4, line=dict(width=1)), line=dict(width=6))

    # cleans the axis titles, but depends on how many subplots there are
    # for i in range(15):
    #     figure['layout'][f'yaxis{i+1 if i else ""}']['title']['text'] = ''

    # figure['layout']['yaxis9']['title']['text'] = 'accuracy'

    # figure['layout']['xaxis']['title']['text'] = ''
    # figure['layout']['xaxis2']['title']['text'] = 'epochs (1024 games/epoch)'
    # figure['layout']['xaxis3']['title']['text'] = ''

    show_plot and figure.show()

    if save:
        os.makedirs("../results", exist_ok=True)
        title = title.replace('<br>', '')
        now = datetime.now().strftime("%Y_%d_%m_%H_%M_%S__")
        # orca requires external installation, can use pip install kaleido instead
        figure.write_image(f"../results/{now}_{str(title)}.png", engine="orca")
