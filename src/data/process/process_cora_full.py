import argparse
import json
import pathlib
import random

import dgl
import torch
from dgl.data import CoraFullDataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "-sp", "--split_proportion", nargs=2, default=[0.7, 0.15], type=float, help="train and test proportions"
)
parser.add_argument("-s", "--seed", default=0, help="seed to random library", type=int)
args = parser.parse_args()

random.seed(args.seed)
assert sum(args.split_proportion) <= 1

OUTPUT_DIR = pathlib.Path("research/data/processed/cora_full/")

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

dataset = CoraFullDataset(raw_dir="research/data/raw/cora_full")

all_matches = {}

edges = ([int(x) for x in dataset[0].edges()[0]], [int(x) for x in dataset[0].edges()[1]])

for a, b in zip(edges[0], edges[1]):
    if a not in all_matches:
        all_matches[a] = set()

    if b not in all_matches:
        all_matches[b] = set()

    all_matches[a].add(b)
    all_matches[b].add(a)

list_of_indexes = list(range(len(dataset[0].nodes())))

random.shuffle(list_of_indexes)

subset_name_to_mask = {}
subset_name_to_mask["train"] = [False] * len(list_of_indexes)
subset_name_to_mask["test"] = [False] * len(list_of_indexes)
subset_name_to_mask["val"] = [False] * len(list_of_indexes)

for i in list_of_indexes[0 : int(args.split_proportion[0] * len(list_of_indexes))]:
    subset_name_to_mask["train"][i] = True

for i in list_of_indexes[
    int(args.split_proportion[0] * len(list_of_indexes)) : int(
        (args.split_proportion[1] + args.split_proportion[0]) * len(list_of_indexes)
    )
]:
    subset_name_to_mask["test"][i] = True

for i in list_of_indexes[int((args.split_proportion[1] + args.split_proportion[0]) * len(list_of_indexes)) :]:
    subset_name_to_mask["val"][i] = True


for subset_name in tqdm(["train", "test", "val"], "processing subsets"):
    csv_str = [",".join([str(a) for a in range(len(dataset[0].ndata["feat"][0]))]) + ",entity_id,record_id\n"]

    matches = {}

    entity_id_to_record_ids = {}

    mask = subset_name_to_mask[subset_name]

    for i in dataset[0].nodes():
        if mask[i]:
            i = int(i)
            entity_id = "id_C-" + str(int(dataset[0].ndata["label"][i]))

            csv_str.append(
                ",".join([str(float(a)) for a in dataset[0].ndata["feat"][i]])
                + ","
                + entity_id
                + ","
                + "id_C-"
                + str(i)
                + "\n"
            )

            if entity_id not in entity_id_to_record_ids:
                entity_id_to_record_ids[entity_id] = []
            entity_id_to_record_ids[entity_id].append("id_C-" + str(i))

    csv_str = "".join(csv_str)

    with (OUTPUT_DIR / (subset_name + "-vectorized.csv")).open("w") as f:
        f.write(csv_str)

    for k in entity_id_to_record_ids:
        record_ids = entity_id_to_record_ids[k]
        for i in record_ids:
            for j in record_ids:
                if j != i:

                    if i not in matches:
                        matches[i] = []

                    matches[i].append(j)

    with (OUTPUT_DIR / (subset_name + "_matches.json")).open("w") as f:
        f.write(json.dumps(matches))
