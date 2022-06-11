import pandas as pd
import plotly
from plotly.offline import download_plotlyjs, plot
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

def scatter(df: pd.DataFrame) -> None:
    # dataset.sort_values(by="Fare")

    fig = px.scatter(df, x="Age", y="Fare",
                     labels={
                         "Age": "Age (years)",
                         "Fare": "Fare (British Pound)"},
                     title="Age to Fare")
    plotly.offline.plot(fig, filename='../plots/scatter_age_fare.html', auto_open=False)

def visualize_missing_data(df: pd.DataFrame) -> None:
    num_missing = df.isna().sum()
    print(num_missing)

    pct_missing = df.isna().mean()
    print(pct_missing)

    fig = px.imshow(df.isna(), color_continuous_scale=["#3f3f3f", "#b7b7b7"])
    plotly.offline.plot(fig, filename="../plots/missing_heatmap.html", auto_open=False)

    fig = px.histogram(df.isna().sum(axis="columns"), nbins=3)
    plotly.offline.plot(fig, filename="../plots/missing_histogram.html", auto_open=False)


def visualize_parameter_distribution(df: pd.DataFrame, param: str) -> None:
    fig = px.histogram(df[param])
    plotly.offline.plot(fig, filename="../plots/parameter_histogram.html", auto_open=False)

def box_plot(df: pd.DataFrame, param: str) -> None:
    fig = px.box(df, y=param)
    plotly.offline.plot(fig, filename="../plots/box_plot.html", auto_open=False)

def visualize_categorical_outliers(df: pd.DataFrame) -> None:
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns
    fig = make_subplots(rows=1,cols=len(non_numeric_cols))
    for i, column in enumerate(non_numeric_cols):
        fig.add_trace(go.Histogram(x=df[column], legendgrouptitle=column), row=1, col=i+1)
    plotly.offline.plot(fig, filename="../plots/categorical_histogram.html", auto_open=False)

if __name__ == "__main__":
    df = pd.read_csv(r"../datasets/train.csv", sep=",", header="infer", names=None, encoding="utf-8")
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    print(df.head(3))

    df.info()

    print(df.kurt(numeric_only=True))

    print(df["Fare"].describe())

    visualize_categorical_outliers(df)


