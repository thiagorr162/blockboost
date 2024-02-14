import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

eval_path = pathlib.Path("research/eval/canopy")
experiments_path = pathlib.Path("research/experiments/canopy")

experiments_path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(eval_path / "final_metrics.csv")
df_test = df[df["datafold"] == "test"]

df_test["rrr"] = df["recall"] * df["RR"]

datasets = df["database"].unique()

for dataset in datasets:
    print(f"Plotting dataset {dataset}")
    df_dataset = df_test[df_test["database"] == dataset]

    # generate plotly plot
    fig = px.scatter(
        df_dataset[df_dataset["database"] == dataset],
        x="recall",
        y="RR",
        hover_data=["t1", "t2"],
        size="rrr",
    )

    fig.write_html(experiments_path / f"plotly-{dataset}.html")
