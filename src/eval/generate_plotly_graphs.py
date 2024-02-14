import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument(
    "--models",
    nargs="+",
    default=["adahash", "tlsh", "klsh", "canopy", "fasthash", "threshold"],
    help="Model name to concatenate evaluations",
)

parser.add_argument(
    "--experiments",
    nargs="+",
    default=["boxplot", "scatter"],
    help="Which type of experiment to run",
)
parser.add_argument(
    "--n_best_models",
    type=int,
    default=100,
)

args = parser.parse_args()

eval_path = pathlib.Path("research/eval/")
output_file_name = "all_tests-" + "_".join(args.models) + ".csv"
full_file_name = eval_path / output_file_name

experiments_path = pathlib.Path("research/experiments_adahash/")
experiments_path.mkdir(parents=True, exist_ok=True)

if not full_file_name.is_file():
    print("\nConcatenating all_tests.jsonl ...")
    all_experiments = []

    for model in tqdm(args.models):

        json_path = eval_path / f"{model}/all_tests.jsonl"

        assert json_path.is_file(), f"{model} does not have all_tests.jsonl"

        with open(json_path, "r") as f:
            for line in f:
                experiment = json.loads(line)

                # Remove seed and commit from hparams... then we can use str(hparams) to groupby experiments
                if "seed" in experiment["hparams"]:
                    experiment["seed"] = experiment["hparams"]["seed"]
                    experiment["hparams"].pop("seed")

                if "commit" in experiment["hparams"]:
                    experiment["commit"] = experiment["hparams"]["commit"]
                    experiment["hparams"].pop("commit")

                if "skip_existing" in experiment["hparams"]:
                    experiment["hparams"].pop("skip_existing")

                all_experiments.append(experiment)

    df = pd.DataFrame(all_experiments)
    df = pd.concat([df, df.hparams.apply(pd.Series)], axis=1)

    df[["database", "nickname"]] = df["database"].str.split("-", 1, expand=True)

    df["precision"] = df["precision"].astype(float)
    df["rrr"] = 2 * (df["recall"] * df["reduction_ratio"]) / (df["recall"] + df["reduction_ratio"])

    if "adahash" in args.models:
        df["model"] = df["model"].where(
            df["model"] != "adahash", df["model"] + "-" + df["lsh_type"] + "-" + df["prediction_type"]
        )

    print("\nSaving...")

    df.to_csv(eval_path / output_file_name)
else:
    print(f"File already exists: {full_file_name} \nLoading")

    df = pd.read_csv(full_file_name)
    df = df.drop("Unnamed: 0", axis=1)


# Make rrr be the first column
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

if "boxplot" in args.experiments:
    # create boxplots for each metric
    datasets_boxplot_path = experiments_path / "viz/plotly/boxplots/dataset_comparison"
    datasets_boxplot_path.mkdir(parents=True, exist_ok=True)

    models_boxplot_path = experiments_path / "viz/plotly/boxplots/model_comparison"
    models_boxplot_path.mkdir(parents=True, exist_ok=True)

    metrics = ["rrr", "recall", "reduction_ratio", "precision", "f_score"]
    models = df["model"].unique()
    dbs = df["database"].unique()

    print("\nGenerating boxplots...")

    pbar = tqdm(metrics)
    for metric in pbar:
        pbar.set_description(f"Processing {metric}")
        for model in models:

            df_model = df[df["model"] == model]
            fig = px.box(
                data_frame=df_model, x="database", y=metric, title=f"Boxplot of {metric} using {model} across datasets"
            )
            fig.write_image(datasets_boxplot_path / f"{metric}-{model}-boxplot_comparing_across_datasets.png")

    for metric in pbar:
        pbar.set_description(f"Processing {metric}")
        for db in dbs:

            df_model = df[df["database"] == db]
            fig = px.box(
                data_frame=df_model, x="model", y=metric, title=f"Boxplot of {metric} in database {db} across models"
            )
            fig.write_image(models_boxplot_path / f"{metric}-{db}-boxplot_comparing_across_models.png")

if "scatter" in args.experiments:
    print("\nGenerating scatter plots...")

    pbar = tqdm(df["database"].unique())
    for db in pbar:

        df_ungrouped = df[df["database"] == db].copy()
        pbar.set_description(f"Processing {db}")

        agg_dict = {}
        cols_to_take_mean = ["recall", "reduction_ratio", "precision", "rrr", "seed", "f_score"]

        for col in df_ungrouped.columns:

            if col in ["hparams", "prediction_type"]:
                continue
            if col in cols_to_take_mean:
                agg_dict[col] = "mean"
            elif col == "output_path":
                agg_dict[col] = lambda x: list(x)
            else:
                agg_dict[col] = "first"

        df_grouped = df_ungrouped.groupby(["hparams", "prediction_type"]).agg(agg_dict)
        df_grouped = df_grouped.reset_index()

        hover_data_options = [
            "seed",
            "recall",
            "reduction_ratio",
            "rrr",
            "f_score",
            "n_buckets",
            "bucket_size",
            "evaluation_time",
            "nickname",
            "proportion_nonmatches",
            "t1",
            "t2",
            "number_of_buckets",
            "shingle_size",
            # "number_of_random_projections",
            "number_of_blocks",
        ]

        fig = px.scatter(
            df_ungrouped,
            x="recall",
            y="reduction_ratio",
            color="model",
            hover_data=hover_data_options,
        )

        # Save plotly plot
        output_plot_name = experiments_path / "viz/plotly/recall_rr"
        output_plot_name.mkdir(parents=True, exist_ok=True)

        output_table_name = experiments_path / "tables/best_models/recall_rr"
        output_table_name.mkdir(parents=True, exist_ok=True)

        fig.write_html(output_plot_name / f"{db}-ungrouped.html")

        fig = px.scatter(
            df_grouped,
            x="recall",
            y="reduction_ratio",
            color="model",
            hover_data=hover_data_options,
        )

        fig.write_html(output_plot_name / f"{db}-grouped.html")

        columns_to_drop = [
            "hparams",
            "evaluation_time",
            "data_type",
            "proportion_matches",
            "n_iter",
            "nonmatches_type",
            "predict_train",
            "lsh_type",
            "full_prediction",
            "nickname",
            "prediction_type",
            # "output_path",
            "subset",
            "seed",
            "commit",
            "skip_existing",
            "database",
            "minimum_hamming_distance",
        ]

        # Save table with best models

        best_rrr_models = df_grouped.sort_values(by="rrr", ascending=False)
        best_rrr_models = best_rrr_models.drop(columns=columns_to_drop, errors="ignore")

        best_rrr_models.iloc[: args.n_best_models].to_csv(
            output_table_name / f"{db}-best_{args.n_best_models}_models.csv"
        )
        best_rrr_models.groupby("model").first().sort_values(by="rrr", ascending=False).to_csv(
            output_table_name / f"{db}-best_experiment_given_model.csv"
        )
        best_rrr_models.groupby("model").first().sort_values(by="rrr", ascending=False).to_json(
            output_table_name / f"{db}-best_experiment_given_model.json",
            orient="records",
        )

        fig = px.scatter(
            df_ungrouped,
            x="recall",
            y="precision",
            color="model",
            hover_data=hover_data_options,
        )

        output_plot_name = experiments_path / "viz/plotly/recall_precision"
        output_plot_name.mkdir(parents=True, exist_ok=True)

        output_table_name = experiments_path / "tables/best_models/recall_precision"
        output_table_name.mkdir(parents=True, exist_ok=True)

        fig.write_html(output_plot_name / f"{db}-ungrouped.html")

        fig = px.scatter(
            df_grouped,
            x="recall",
            y="precision",
            color="model",
            hover_data=hover_data_options,
        )

        fig.write_html(output_plot_name / f"{db}-grouped.html")

        # Save table with best models

        best_f_models = df_grouped.sort_values(by="f_score", ascending=False)
        best_f_models = best_f_models.drop(columns=columns_to_drop, errors="ignore")

        best_f_models.iloc[: args.n_best_models].to_csv(
            output_table_name / f"{db}-best_{args.n_best_models}_models.csv"
        )
        best_f_models.groupby("model").first().sort_values(by="f_score", ascending=False).to_csv(
            output_table_name / f"{db}-best_experiment_given_model.csv"
        )
        best_f_models.groupby("model").first().sort_values(by="f_score", ascending=False).to_json(
            output_table_name / f"{db}-best_experiment_given_model.json", orient="records"
        )
