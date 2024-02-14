import argparse
import csv
import itertools
import json
import pathlib

parser = argparse.ArgumentParser()

parser.add_argument(
    "-db",
    "--database",
    default="RLDATA500",
    help="Database to block.",
    type=str,
)

parser.add_argument(
    "-s",
    "--size",
    default=2,
    help="Use the combinations with size s of the columns as hashes.",
    type=int,
)

parser.add_argument(
    "--sort_hashes",
    action="store_true",
)

args = parser.parse_args()

data_dir = pathlib.Path(f"research/data/processed/{args.database}")

data_folds_to_predict = ["test", "val"]

model_dir = pathlib.Path(f"research/models/fasthash/{args.database}-{args.size}-{args.sort_hashes}")
model_dir.mkdir(parents=True, exist_ok=True)

with (model_dir / "hparams.json").open("w") as f:
    hparams = vars(args)
    hparams["model"] = "fasthash"
    f.write(json.dumps(hparams, indent=4))

for data_fold in data_folds_to_predict:
    data_path = data_dir / (data_fold + "-textual.csv")
    f = data_path.open("r")
    data = csv.reader(f, delimiter=",")

    hash_to_ei = {}

    for row in list(data)[1:]:
        features = [x for x in row[1:-2] if len(x) > 0]
        hashes = itertools.combinations(features, args.size)
        record_id = row[-1]
        ei_to_hashes = hashes
        for h in hashes:
            if args.sort_hashes:
                h = tuple(sorted(h))
            if h not in hash_to_ei:
                hash_to_ei[h] = []
            hash_to_ei[h].append(record_id)
    f.close()

    list_of_clusters = []
    for h in hash_to_ei:
        list_of_clusters += [[hash_to_ei[h]]]

    with (model_dir / f"{data_fold}-full_prediction.json").open("w") as g:
        g.write(json.dumps(list_of_clusters))
