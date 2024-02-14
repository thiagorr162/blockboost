import argparse
import itertools
import json
import math
import pathlib
import subprocess

import numpy as np
import pandas as pd
from sklearn import preprocessing

from src.models.canopy.model import Canopy


def get_git_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


parser = argparse.ArgumentParser()

parser.add_argument(
    "-db",
    "--database",
    default="RLDATA10000-vectorize_minhash-amber_ox-1",
    help="Database to block.",
    type=str,
)
parser.add_argument(
    "-dt",
    "--data_type",
    choices=["textual", "vectorized"],
    default="vectorized",
    help="Data type.",
    type=str,
)
parser.add_argument(
    "-dist",
    "--distance_type",
    default="euclidean",
    help="Distance function.",
    type=str,
)
parser.add_argument(
    "-t1",
    default=1,
    help="canopy external threshold",
    type=float,
)
parser.add_argument(
    "-t2",
    default=1,
    help="canopy internal threshold = t2*t1, t2 in (0,1)",
    type=float,
)
parser.add_argument(
    "--predict_train",
    action="store_true",
    help="Make predictions for train data.",
)
parser.add_argument(
    "-s",
    "--seed",
    default=0,
    type=int,
)

args = parser.parse_args()

data_dir = pathlib.Path(f"research/data/processed/{args.database}")
model_dir = pathlib.Path(
    f"research/models/canopy/{args.database}/seed_{args.seed}-t1_{args.t1}-t2_{args.t2}-dist_{args.distance_type}-datatype_{args.data_type}"
)
model_dir.mkdir(parents=True, exist_ok=True)

# calculating canopy centers
df = pd.read_csv(data_dir / f"train-{args.data_type}.csv")

df_features = df.iloc[:, :-2]

if args.data_type == "vectorized":

    min_train = df_features.min().min()
    max_train = df_features.max().max()
    df_features = (df_features - min_train) / (max_train - min_train)
    df_features = df_features / np.sqrt(df_features.shape[1])

arr = df_features.values

gc = Canopy(arr, distance_type=args.distance_type, seed=args.seed)
gc.setThreshold(args.t1, args.t2 * args.t1)
canopies = gc.clustering()
canopies_centers = [el[0] for el in canopies]

# predict blocks

data_folds_to_predict = ["test", "val"]

if args.predict_train:
    data_folds_to_predict.append("train")

for data_fold in data_folds_to_predict:

    print(f"Predicting {data_fold} data...")

    output_path = model_dir / f"{data_fold}-full_prediction.json"

    df = pd.read_csv(data_dir / f"{data_fold}-{args.data_type}.csv")
    df_features = df.iloc[:, :-2]

    if args.data_type == "vectorized":
        df_features = (df_features - min_train) / (max_train - min_train)
        df_features = df_features / np.sqrt(df_features.shape[1])

    arr = df_features.values

    df_features = pd.DataFrame(arr, columns=df_features.columns, index=df_features.index)
    df_features["entity_id"] = df["entity_id"]
    df_features["record_id"] = df["record_id"]

    full_prediction = []

    print(f"Number of centers: {len(canopies_centers)}")
    for i, center in enumerate(canopies_centers):

        print(f"Current center: {i}")
        dists = df_features.iloc[:, :-2].apply(gc.distance, vec2=center, axis=1)
        cluster = list(df_features["record_id"][dists < args.t1])
        full_prediction.append([cluster])

    print(f"output_path={output_path}")

    with output_path.open("w") as out_file:
        json.dump(full_prediction, out_file)

    with (model_dir / "hparams.json").open("w") as f:
        hparams = {}
        hparams["model"] = "canopy"
        hparams["distance"] = args.distance_type
        hparams["commit"] = get_git_revision()
        hparams["database"] = args.database
        hparams["data_type"] = args.data_type
        hparams["t1"] = args.t1
        hparams["t2"] = args.t2
        hparams["seed"] = args.seed
        f.write(json.dumps(hparams, indent=True))
