import pandas as pd
import plotly
from plotly.offline import download_plotlyjs, plot
import plotly.graph_objs as go
import plotly.express as px

def visualize(dataset: pd.DataFrame) -> None:
    # dataset.sort_values(by="Fare")

    fig = px.scatter(dataset, x="Age", y="Fare",
                     labels={
                         "Age": "Age (years)",
                         "Fare": "Fare (British Pound)"},
                     title="Age to Fare")
    plotly.offline.plot(fig, filename='scatter_age_fare.html', auto_open=False)