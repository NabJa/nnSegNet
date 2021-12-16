from pathlib import Path
from collections import defaultdict

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.express as px
import pandas as pd

from brats21 import utils as bu
from brats21 import evaluation as valid

TRAINED_MODEL_PATH = Path("/sc-scratch/sc-scratch-gbm-radiomics/nnUNet_trained_models")
MODEL_PREDICTION_PATH = Path(
    "/sc-scratch/sc-scratch-gbm-radiomics/task1_evaluation/output"
)

TEXT_STYLE = {"textAlign": "center", "color": "#191970"}


####
# Training Plot
####
performance = pd.DataFrame(bu.get_model_histroy(TRAINED_MODEL_PATH))
performance = performance.groupby(["name", "iteration"]).mean().reset_index()

training_dropdown = dcc.Dropdown(
    id="training_dropdown",
    options=[
        {"label": "Dice", "value": "valid_metric"},
        {"label": "Validation Loss", "value": "valid_loss"},
        {"label": "Training Loss", "value": "train_loss"},
    ],
    value="valid_metric",
    clearable=False,
)

train_smooth_slider = html.Div(
    [
        html.Div("Smooth"),
        dcc.Slider(id="train_smooth_slider", min=1, max=30, step=1, value=5),
    ],
    style={"padding": 5, "margin": 5},
)


train_history = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Training History"),
                dcc.Graph(id="train_plot"),
                training_dropdown,
                train_smooth_slider,
            ]
        )
    ],
    style={"margin": 10},
)

####
# Metric Plot
####

NAMING = {
    "nnUNetTrainerSegNetTransformer": "Transformer",
    "nnUNetTrainerSegNetMHSA": "MHSA",
    "nnUNetTrainerSegResNetEnc": "ResNet-Encoder",
    "nnUNetTrainerSegResNetDec": "ResNet-Decoder",
    "nnUNetTrainerV2BraTSSegnet": "SegNet",
    "nnUNetTrainerV2BraTSRegions": "UNet",
}

summarys = [
    valid.Summary(path=path, name=path.parents[1].name)
    for path in MODEL_PREDICTION_PATH.rglob("new_summary.json")
]


def get_summary_df(summarys, metric):
    df = defaultdict(list)
    for summary in summarys:
        for result in summary.image:
            for c in ["Edema", "Enhancing", "Necrotic"]:
                df["Model"].append(NAMING[summary.name])
                df["Class"].append(c)
                df["Value"].append(result.__getattribute__(metric)[c])
    return pd.DataFrame(df)


metric_dropdown = dcc.Dropdown(
    id="metric_dropdown",
    options=[
        {"label": "Dice", "value": "dice"},
        {"label": "Jaccard", "value": "jaccard"},
        {"label": "Precision", "value": "precision"},
        {"label": "Recall", "value": "recall"},
        {"label": "Hausdorff", "value": "hausdorff"},
        {"label": "Hausdorff 95% CI", "value": "hausdorff95"},
        {"label": "False Discovery Rate", "value": "fdr"},
        {"label": "False Positive Rate", "value": "fpr"},
        {"label": "False Negative Rate", "value": "fnr"},
        {"label": "True Negative Rate", "value": "tnr"},
        {"label": "False Omission Rate", "value": "false_omission"},
        {"label": "Negative Predictive Value", "value": "npv"},
        {"label": "Total Positives Test", "value": "total_positive_test"},
        {"label": "Total Positives Reference", "value": "total_positives_reference"},
    ],
    value="dice",
    clearable=False,
)


validation_performance = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Validation Performance"),
                dcc.Graph(id="valid_fig"),
                metric_dropdown,
            ]
        )
    ],
    style={"margin": 10},
)


####
# App Main
####
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    dbc.Row([dbc.Col(train_history), dbc.Col(validation_performance)])
)


####
# Callbacks
####


@app.callback(
    Output(component_id="valid_fig", component_property="figure"),
    Input(component_id="metric_dropdown", component_property="value"),
)
def select_metric(value):
    df = get_summary_df(summarys, value)
    metric_plot = px.box(df, x="Model", y="Value", color="Class")
    return metric_plot


@app.callback(
    Output(component_id="train_plot", component_property="figure"),
    Input(component_id="training_dropdown", component_property="value"),
    Input(component_id="train_smooth_slider", component_property="value"),
)
def update_train_plot(dropdown, slider):

    df = performance.copy()

    df.loc[:, dropdown] = bu.mean_smooth(df.loc[:, dropdown], slider)

    train_plot = px.line(
        df,
        x="iteration",
        y=dropdown,
        color="name",
        labels={
            "name": "Model",
            "valid_metric": "Dice Score",
            "train_loss": "Training Loss",
            "valid_loss": "Validation Loss",
            "iteration": "Epoch",
        },
    )
    return train_plot


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0")
