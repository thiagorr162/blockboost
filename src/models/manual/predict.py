import argparse
import csv
import itertools
import json
import pathlib

parser = argparse.ArgumentParser()

parser.add_argument(
    "-db",
    "--database",
    default="musicbrainz_20",
    help="Database to block.",
    type=str,
)

parser.add_argument(
    "-f",
    "--features",
    default=["number", "length"],
    help="Features to be used as hashes.",
    nargs="+",
    type=str,
)

args = parser.parse_args()

data_dir = pathlib.Path(f"research/data/processed/{args.database}")

data_folds_to_predict = ["test", "val"]
args.size = 2
model_dir = pathlib.Path(f"research/models/manual/{args.database}-{'-'.join(args.features)}")
model_dir.mkdir(parents=True, exist_ok=True)

with (model_dir / "hparams.json").open("w") as f:
    hparams = vars(args)
    hparams["model"] = "manual"
    f.write(json.dumps(hparams, indent=4))

for data_fold in data_folds_to_predict:
    data_path = data_dir / (data_fold + "-textual.csv")
    f = data_path.open("r")
    data = csv.reader(f, delimiter=",")

    hash_to_ei = {}
    data= list(data)
    header = data[0]
    f_2_i = {}
    for i in range(len(header)):
        f_2_i[header[i]] = i

    f_ind = [f_2_i[f] for f in args.features]
    for row in (data)[1:]:
        row = list(row)
        print(row)
        print()
        hashes = [";".join([row[i] for i in f_ind ])]
        #hashes = itertools.combinations(features, args.size)
        print(hashes)
        record_id = row[-1]
        ei_to_hashes = hashes
        for h in hashes:
            if h not in hash_to_ei:
                hash_to_ei[h] = []
            hash_to_ei[h].append(record_id)
    f.close()

    list_of_clusters = []
    for h in hash_to_ei:
        list_of_clusters += [[hash_to_ei[h]]]

    with (model_dir / f"{data_fold}-full_prediction.json").open("w") as g:
        g.write(json.dumps(list_of_clusters))
