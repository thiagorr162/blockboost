# flake8: noqa: E402

import argparse
import json
import pathlib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--prediction_path",
    type=str,
)
parser.add_argument(
    "-s",
    "--skip_existing",
    action="store_true",
)


args = parser.parse_args()

PROCESSED_DATA_PATH = pathlib.Path("research/data/processed")

prediction_path = pathlib.Path(args.prediction_path)

eval_folder = pathlib.Path(str(prediction_path).replace("/models/", "/eval/")).parent
eval_folder.mkdir(parents=True, exist_ok=True)
output_path = eval_folder / prediction_path.name

if output_path.exists() and args.skip_existing:
    print(f"skipping existing evaluation: {output_path}")
    exit(0)

from itertools import combinations
from time import time

import pandas as pd
from numpy import argmax
from tqdm import tqdm

assert str(prediction_path).endswith("prediction.json")

with (prediction_path.parent / "hparams.json").open("r") as f:
    hparams = json.loads(f.read())


subset = prediction_path.name.split("-")[0]
prediction_type = prediction_path.name.split("-")[1].split("_")[0]

matches = json.load(open(PROCESSED_DATA_PATH / hparams["database"] / f"{subset}_matches.json", "r"))
predictions = json.load(open(prediction_path, "r"))

assert prediction_type in ["full", "transitive"]

# add backwards compatibility to eval
if not any([k.startswith("id_") for k in matches]):
    old_matches = matches
    matches = {}

    for k in old_matches:
        matches["id_C-" + str(k)] = ["id_C-" + str(v) for v in old_matches[k]]

    if prediction_type == "full":
        old_predictions = predictions
        predictions = [
            [["id_C-" + str(x) for x in cluster] for cluster in list_of_clusters]
            for list_of_clusters in old_predictions
        ]
    else:
        old_predictions = predictions
        predictions = [["id_C-" + str(x) for x in cluster] for cluster in old_predictions]

eval_results = None

start_eval = time()

# one table problem
if all([k.startswith("id_C") for k in matches]):

    if prediction_type == "transitive":

        ei_to_cluster_id = {}  # ei = Entity Id
        cluster_id = 0

        # theoretical comparisons
        len_of_prediction_pairs = 0

        for cluster in predictions:
            for ei in cluster:
                ei_to_cluster_id[ei] = cluster_id
            cluster_id += 1
            len_of_prediction_pairs += len(cluster) * (len(cluster) - 1) / 2

        len_of_matches_pairs = 0
        len_of_true_predictions_pairs = 0

        for record_id in matches:
            int_record_id = record_id
            main_cluster_id = ei_to_cluster_id[int_record_id]
            matches_ids = {k for k in matches[record_id] if k > int_record_id}

            len_of_matches_pairs += len(matches_ids)
            len_of_true_predictions_pairs += sum([ei_to_cluster_id[ei] == main_cluster_id for ei in matches_ids])

        n = len(matches)
        recall = (len_of_matches_pairs == 0) or len_of_true_predictions_pairs / len_of_matches_pairs
        reduction_ratio = 1 - len_of_prediction_pairs / (n * (n - 1) / 2)
        precision = (len_of_prediction_pairs == 0) or len_of_true_predictions_pairs / len_of_prediction_pairs

    elif prediction_type == "full":
        prediction_pairs = {(a, b) for x in predictions for y in x for a in y for b in y if a > b}

        matches_pairs = set()
        for a in matches:
            for b in matches[a]:
                a = str(a)
                b = str(b)
                if a > b:
                    matches_pairs.add((a, b))

        true_predictions = matches_pairs.intersection(prediction_pairs)
        n = len(matches)

        recall = (len(matches_pairs) == 0) or len(true_predictions) / len(matches_pairs)
        reduction_ratio = 1 - len(prediction_pairs) / (n * (n - 1) / 2)
        precision = (len(prediction_pairs) == 0) or len(true_predictions) / len(prediction_pairs)

else:

    table_ids = list(set(k[:4] for k in matches))

    table_ids_comb = list(combinations(table_ids, 2))

    print(table_ids)
    print(table_ids_comb)

    if prediction_type == "transitive":

        ei_to_cluster_id = {}  # ei = Entity Id
        cluster_id = 0

        # theoretical comparisons
        len_of_prediction_pairs = 0

        for cluster in predictions:
            for ei in cluster:
                ei_to_cluster_id[ei] = cluster_id
            cluster_id += 1

            for id_A, id_B in table_ids_comb:
                cluster_A = [record_id for record_id in cluster if record_id.startswith(id_A)]
                cluster_B = [record_id for record_id in cluster if record_id.startswith(id_B)]

                len_of_prediction_pairs += len(cluster_A) * len(cluster_B)

        len_of_matches_pairs = 0
        len_of_true_predictions_pairs = 0

        for record_id in matches:
            int_record_id = record_id
            main_cluster_id = ei_to_cluster_id[int_record_id]
            matches_ids = {k for k in matches[record_id] if k > int_record_id}

            len_of_matches_pairs += len(matches_ids)
            len_of_true_predictions_pairs += sum([ei_to_cluster_id[ei] == main_cluster_id for ei in matches_ids])

        n = len(matches)

        n_pairs = 1

        for table_id in table_ids:
            n_T = len([x for x in matches if x.startswith(table_id)])
            n_pairs *= n_T
        #n_A = len([x for x in matches if x.startswith("id_A")])
        #n_B = len([x for x in matches if x.startswith("id_B")])

        recall = (len_of_matches_pairs == 0) or len_of_true_predictions_pairs / len_of_matches_pairs
        reduction_ratio = 1 - len_of_prediction_pairs / (n_pairs)
        precision = (len_of_prediction_pairs == 1) or len_of_true_predictions_pairs / len_of_prediction_pairs

    elif prediction_type == "full":
        #assert set([k[:4] for k in matches]) == set(["id_A", "id_B"])

        prediction_pairs = {
            (a, b)
            for x in predictions
            for y in x
            for a in y
            for b in y
            if a > b and a[:4] != b[:4]
            # a.startswith("id_B") and b.startswith("id_A")
        }

        matches_pairs = set()
        for a in matches:
            for b in matches[a]:
                a = str(a)
                b = str(b)
                if a > b:
                    matches_pairs.add((a, b))

        true_predictions = matches_pairs.intersection(prediction_pairs)
        n = len(matches)
        n_pairs = 1

        for table_id in table_ids:
            n_T = len([x for x in matches if x.startswith(table_id)])
            n_pairs *= n_T

        recall = (len(matches_pairs) == 0) or len(true_predictions) / len(matches_pairs)
        reduction_ratio = 1 - len(prediction_pairs) / (n_pairs)
        precision = (len(prediction_pairs) == 0) or len(true_predictions) / len(prediction_pairs)


f_score = 0
if recall + precision > 0:
    f_score = 2 * recall * precision / (recall + precision)


eval_results = {
    "recall": recall,
    "reduction_ratio": reduction_ratio,
    "precision": precision,
    "f_score": f_score,
    "hparams": hparams,
    "output_path": str(output_path),
}

end_eval = time()

eval_results["prediction_type"] = prediction_type
eval_results["subset"] = subset
eval_results["evaluation_time"] = end_eval - start_eval

with output_path.open("w") as f:
    f.write(json.dumps(eval_results, indent=4))
