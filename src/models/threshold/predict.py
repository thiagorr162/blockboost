import argparse
import csv
import itertools
import json
import pathlib

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    "-db",
    "--database",
    default="RLDATA10000-vectorize_minhash-amber_ox-1",
    help="Database to block.",
    type=str,
)

parser.add_argument(
    "-m",
    "--metric",
    default="hamming",
    help="Metric",
    type=str,
)

parser.add_argument(
    "-pm",
    "--proportion_of_matches",
    default=1.0,
    help="Given pm, select the smallest threshold t such that the number of matches with distance<=t is greater than or equal to pm*estimated_number_of_matches.",
    type=float,
)

args = parser.parse_args()

data_dir = pathlib.Path(f"research/data/processed/{args.database}")

data_folds_to_predict = ["test", "val"]

model_dir = pathlib.Path(f"research/models/threshold/{args.database}/{args.metric}-{args.proportion_of_matches}")
model_dir.mkdir(parents=True, exist_ok=True)

with (model_dir / "hparams.json").open("w") as f:
    hparams = vars(args)
    hparams["model"] = "threshold"
    f.write(json.dumps(hparams, indent=4))


for data_fold in data_folds_to_predict:
    # Compute n_matches
    matches = json.load((data_dir / f"{data_fold}_matches.json").open("r"))
    n_matches = 0
    for m in matches:
        n_matches += len(matches[m])

    # Load features
    data_path = data_dir / (data_fold + "-vectorized.csv")
    f = data_path.open("r")
    data = csv.reader(f, delimiter=",")
    list_data = list(data)[1:]
    _features = [[int(x) for x in row[1:-2]] for row in list_data]
    record_ids = [row[-1] for row in list_data]
    features = np.array(_features)
    f.close()

    # Compute hamming distance
    hamming = np.count_nonzero(features - features[:, None], axis=2)

    # Construct buckets
    m = np.max(hamming)
    buckets = [0] * (m + 1)

    buckets[0] -= hamming.shape[0]

    for k in hamming.ravel():
        buckets[k] += 1

    selected_d = -1
    for i in range(m):
        buckets[i + 1] += buckets[i]
        if buckets[i + 1] >= n_matches * args.proportion_of_matches and selected_d == -1:
            selected_d = i + 1

    clusters = [
        [[record_ids[int(v[0])], record_ids[int(v[1])]]] for v in np.argwhere(hamming <= selected_d) if v[0] != v[1]
    ]

    with (model_dir / f"{data_fold}-full_prediction.json").open("w") as g:
        g.write(json.dumps(clusters))
