import argparse
import json
import pathlib
import pickle

import numpy as np
import pandas as pd
import pyreadr

parser = argparse.ArgumentParser()
parser.add_argument(
    "-sp",
    "--split_proportion",
    nargs=2,
    default=[0.70, 0.15],
    help="Split proportion of train and test used to generate train, test and val sets.",
    type=float,
)
parser.add_argument(
    "-s",
    "--seed",
    default=42,
    help="Seed to random number generator.",
    type=int,
)

args = parser.parse_args()

np.random.seed(args.seed)

output_dir = pathlib.Path("research/data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_dir = output_dir / "restaurant"
output_dir.mkdir(parents=True, exist_ok=True)

raw_dir = pathlib.Path("research/data/raw/restaurant")

restaurant_data = raw_dir / "restaurant.csv"

df = pd.read_csv(restaurant_data)
df.columns = ["name", "address", "location", "cuisine", "id"]

df["id"] = df["id"].str.replace("'", "").astype(str)

dup_ids = df[df["id"].duplicated()]["id"].values
entity_ids = df[~df["id"].duplicated(keep=False)]["id"].values

np.random.shuffle(dup_ids)
np.random.shuffle(entity_ids)

train, test = args.split_proportion[0], args.split_proportion[1]

idx_train, idx_test = int(len(dup_ids) * train), int(len(dup_ids) * test)
train_ids, test_ids, val_ids = (
    dup_ids[:idx_train],
    dup_ids[idx_train : idx_train + idx_test],
    dup_ids[idx_train + idx_test :],
)

# Add the unique ids
idx_train, idx_test = int(len(entity_ids) * train), int(len(entity_ids) * test)
train_ids = np.concatenate((train_ids, entity_ids[:idx_train]))
test_ids = np.concatenate((test_ids, entity_ids[idx_train : idx_train + idx_test]))
val_ids = np.concatenate((val_ids, entity_ids[idx_train + idx_test :]))

df = df.rename(columns={"id": "entity_id"})

df["record_id"] = df.index.astype(str)

df_train = df[df["entity_id"].isin(train_ids)].copy()
df_test = df[df["entity_id"].isin(test_ids)].copy()
df_val = df[df["entity_id"].isin(val_ids)].copy()

df_train.loc[:, ("entity_id")] = "id_C-" + df_train["entity_id"].astype(str)
df_train.loc[:, ("record_id")] = "id_C-" + df_train["record_id"].astype(str)

df_test.loc[:, ("entity_id")] = "id_C-" + df_test["entity_id"].astype(str)
df_test.loc[:, ("record_id")] = "id_C-" + df_test["record_id"].astype(str)

df_val.loc[:, ("record_id")] = "id_C-" + df_val["record_id"].astype(str)
df_val.loc[:, ("record_id")] = "id_C-" + df_val["record_id"].astype(str)

df_train.to_csv(output_dir / "train-textual.csv", index=False)
df_test.to_csv(output_dir / "test-textual.csv", index=False)
df_val.to_csv(output_dir / "val-textual.csv", index=False)

# Save a json mapping each record_id with the record_ids with the same entity_id
datafold_names = ["train", "test", "val"]
datafolds = [df_train, df_test, df_val]
for datafold_name, datafold in zip(datafold_names, datafolds):
    datafold_matches = {}
    for record_id, entity_id in zip(datafold["record_id"], datafold["entity_id"]):
        datafold_matches[record_id] = datafold["record_id"][datafold["entity_id"] == entity_id].tolist()
        datafold_matches[record_id].remove(record_id)
    json.dump(datafold_matches, open(output_dir / f"{datafold_name}_matches.json", "w"), indent=1)
