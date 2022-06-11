import pandas as pd
import plotly
from plotly.offline import download_plotlyjs, plot
import plotly.graph_objs as go
import plotly.express as px

def scatter(df: pd.DataFrame) -> None:
    # dataset.sort_values(by="Fare")

    fig = px.scatter(df, x="Age", y="Fare",
                     labels={
                         "Age": "Age (years)",
                         "Fare": "Fare (British Pound)"},
                     title="Age to Fare")
    plotly.offline.plot(fig, filename='../plots/scatter_age_fare.html', auto_open=False)

def visualize_missing_data(df: pd.DataFrame) -> None:
    fig = px.imshow(df.isna(), color_continuous_scale=["#3f3f3f", "#b7b7b7"])
    plotly.offline.plot(fig, filename="../plots/missing_heatmap.html", auto_open=False)

    fig = px.histogram(df.isna().sum(axis="columns"), nbins=3)

    plotly.offline.plot(fig, filename="../plots/missing_histogram.html", auto_open=False)