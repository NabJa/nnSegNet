import pandas as pd
import plotly.graph_objects as go
import numpy as np

def meanstd(df, name: str, group: str):
    
    # Group dataframe and calculate mean and std
    grouped_df = df[df.name == name].groupby(group, as_index=False)
    mean, std = grouped_df.mean(), grouped_df.std()
    
    # Add name convention to mean and std dataframes
    mean.rename(mapper=lambda x: f"{x}_mean" if x != group else group, axis=1, inplace=True)
    std.rename(mapper=lambda x:  f"{x}_std" if x != group else "redundant", axis=1, inplace=True)

    # Concatenate mean and std dataframes
    df = pd.concat([mean, std], axis=1)
    
    # Add name column
    df["Name"] = name
    
    # Remove redundant second grouping column
    df.drop("redundant", axis=1, inplace=True)
    
    return df

def get_mean_std_performance(df):
    df = [meanstd(df, name, "iteration") for name in df.name.unique()]
    return pd.concat(df, axis=0)


def random_color(alpha=1.0):
    colors = [np.random.randint(0, 255) for _ in range(3)]
    return "rgb({}, {}, {})".format(*colors), "rgba({}, {}, {}, {})".format(*colors, alpha)

def plotly_line(df, x: str, avg: str, std: str, name: str):
    _, err_color = random_color(0.8)
    return go.Scatter(
            name=name,
            x=df[x],
            y=df[avg],
            mode='lines',
            line=dict(color=err_color),
        )

def cont_err_plot(df):
    df = get_mean_std_performance(df)
    fig = go.Figure()
    for name in df.Name.unique():
        fig.add_trace(
            plotly_line(df[df.Name == name], x="iteration", avg="valid_metric_mean", std="valid_metric_std", name=name)
        )
    return fig