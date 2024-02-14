import argparse
import json
import pathlib

import pandas as pd
import plotly.express as px
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    "-m",
    "--model",
    default="adahash",
    help="Model name to concatenate evaluations",
    type=str,
)
parser.add_argument(
    "-tr",
    "--include_train",
    help="Include train evaluations",
    action="store_true",
)
parser.add_argument(
    "-c",
    "--concatenate_metrics",
    help="Create file that concatenates metrics across all datasets",
    action="store_true",
)


args = parser.parse_args()

model_path = pathlib.Path(f"research/eval/{args.model}")
experiments_path = pathlib.Path(f"research/experiments/{args.model}")

experiments_path.mkdir(parents=True, exist_ok=True)

data_folds = ["test", "val"]

if args.include_train:
    data_folds.append("train")

if args.concatenate_metrics:

    df_full = pd.DataFrame([])

    for data_fold in data_folds:

        print(f"Concatenating {data_fold} results")
        print("----------------------------------")

        all_models = model_path.rglob(f"**/{data_fold}-*_prediction.json")

        for model in tqdm(all_models):

            with open(model) as f:
                data = json.load(f)

            df_params = pd.DataFrame([data["hparams"]])
            df_params["datafold"] = data["subset"]

            if "full_prediction" in str(model.stem):
                df_params["prediction_type"] = "full"
            elif "transitive_prediction" in str(model.stem):
                df_params["prediction_type"] = "transitive"
            else:
                df_params["prediction_type"] = "None"

            df_metrics = pd.DataFrame([data["recall"], data["reduction_ratio"], data["precision"]]).T
            df_metrics.columns = ["recall", "RR", "precision"]

            df_model = pd.concat([df_params, df_metrics], axis=1)
            df_full = pd.concat([df_model, df_full], axis=0)

    df_full.to_csv(model_path / "final_metrics.csv", index=False)

else:

    df_full = pd.read_csv(model_path / "final_metrics.csv")

df_full["HM"] = 2 * (df_full["recall"] * df_full["RR"]) / (df_full["recall"] + df_full["RR"])
df_full["RRR"] = df_full["recall"] * df_full["RR"]

datasets = df_full["database"].unique()

for dataset in datasets:

    print(f"Plotting dataset {dataset}")
    df_dataset = df_full[df_full["database"] == dataset]

    fig = px.scatter(
        df_dataset,
        x="recall",
        y="RR",
        color="distance",
        hover_data=["t1", "t2"],
        size="RRR",
        symbol="prediction_type",
    )

    fig.write_html(experiments_path / f"plotly-{dataset}.html")
