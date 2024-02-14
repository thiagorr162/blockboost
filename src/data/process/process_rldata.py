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

output_dir_500 = output_dir / "RLDATA500"
output_dir_500.mkdir(parents=True, exist_ok=True)

output_dir_10000 = output_dir / "RLDATA10000"
output_dir_10000.mkdir(parents=True, exist_ok=True)

raw_dir = pathlib.Path("research/data/raw/rldata")

raw_rldata10000 = raw_dir / "rldata10000.rda"
raw_rldata500 = raw_dir / "rldata500.rda"

rda_rldata10000 = pyreadr.read_r(raw_rldata10000)
rda_rldata500 = pyreadr.read_r(raw_rldata500)

print(rda_rldata500)

X_rldata10000 = rda_rldata10000["RLdata10000"]
Y_rldata10000 = rda_rldata10000["identity.RLdata10000"]

X_rldata500 = rda_rldata500["RLdata500"]
Y_rldata500 = rda_rldata500["identity.RLdata500"]

colnames = ["first_name_1", "first_name_2", "last_name_1", "last_name_2", "birth_year", "birth_month", "birth_day"]
X_rldata500.columns = colnames
X_rldata10000.columns = colnames

# fill na with empty strings and create full name variable
name_cols = ["first_name_1", "first_name_2", "last_name_1", "last_name_2"]

for col in name_cols:
    X_rldata500[col] = X_rldata500[col].fillna("")
    X_rldata10000[col] = X_rldata10000[col].fillna("")

    X_rldata500["full_name"] = X_rldata500[name_cols].apply(lambda row: " ".join(row.values.astype(str)), axis=1)
    X_rldata10000["full_name"] = X_rldata10000[name_cols].apply(lambda row: " ".join(row.values.astype(str)), axis=1)

    X_rldata500["full_name"] = X_rldata500["full_name"].str.replace("  ", " ")
    X_rldata10000["full_name"] = X_rldata10000["full_name"].str.replace("  ", " ")

    X_rldata500["full_name"] = X_rldata500["full_name"].str.replace(" $", "", regex=True)
    X_rldata10000["full_name"] = X_rldata10000["full_name"].str.replace(" $", "", regex=True)

X_rldata500["entity_id"] = "id_C-" + Y_rldata500["identity.RLdata500"].astype(int).astype(str)
X_rldata10000["entity_id"] = "id_C-" + Y_rldata10000["identity.RLdata10000"].astype(int).astype(str)

del Y_rldata500, Y_rldata10000

duplicated_ids_500 = X_rldata500[X_rldata500["entity_id"].duplicated()]["entity_id"].values
duplicated_ids_10000 = X_rldata10000[X_rldata10000["entity_id"].duplicated()]["entity_id"].values

# unique ids are the negation (~) of duplicated ones
entity_ids_500 = X_rldata500[~X_rldata500["entity_id"].duplicated(keep=False)]["entity_id"].values
entity_ids_10000 = X_rldata10000[~X_rldata10000["entity_id"].duplicated(keep=False)]["entity_id"].values

np.random.shuffle(duplicated_ids_500)
np.random.shuffle(duplicated_ids_10000)
np.random.shuffle(entity_ids_500)
np.random.shuffle(entity_ids_10000)

# get train and test proportions
TRAIN, TEST = args.split_proportion[0], args.split_proportion[1]

# calculate train, test and val ids that come from duplicated ids in RLDATA500
idx_train, idx_test = int(len(duplicated_ids_500) * TRAIN), int(len(duplicated_ids_500) * TEST)
train_ids, test_ids, val_ids = (
    duplicated_ids_500[:idx_train],
    duplicated_ids_500[idx_train : idx_train + idx_test],
    duplicated_ids_500[idx_train + idx_test :],
)

# Add the unique ids
idx_train, idx_test = int(len(entity_ids_500) * TRAIN), int(len(entity_ids_500) * TEST)
train_ids = np.concatenate((train_ids, entity_ids_500[:idx_train]))
test_ids = np.concatenate((test_ids, entity_ids_500[idx_train : idx_train + idx_test]))
val_ids = np.concatenate((val_ids, entity_ids_500[idx_train + idx_test :]))

X_rldata500["record_id"] = "id_C-" + X_rldata500.index.astype(str)

# For each set, only keep the ids that are relevant to the set
# i.e. for training set, keep only the rows in the full data that contain ids in train_ids
X_rldata500_train = X_rldata500[X_rldata500["entity_id"].isin(train_ids)]
X_rldata500_test = X_rldata500[X_rldata500["entity_id"].isin(test_ids)]
X_rldata500_val = X_rldata500[X_rldata500["entity_id"].isin(val_ids)]

X_rldata500_train.to_csv(output_dir_500 / "train-textual.csv", index=False)
X_rldata500_test.to_csv(output_dir_500 / "test-textual.csv", index=False)
X_rldata500_val.to_csv(output_dir_500 / "val-textual.csv", index=False)

# Save a json mapping each record_id with the record_ids with the same entity_id
datafold_names = ["train", "test", "val"]
datafolds = [X_rldata500_train, X_rldata500_test, X_rldata500_val]
for datafold_name, datafold in zip(datafold_names, datafolds):
    rldata500_datafold_matches = {}
    for record_id, entity_id in zip(datafold["record_id"], datafold["entity_id"]):
        rldata500_datafold_matches[record_id] = datafold["record_id"][datafold["entity_id"] == entity_id].tolist()
        rldata500_datafold_matches[record_id].remove(record_id)
    json.dump(rldata500_datafold_matches, open(output_dir_500 / f"{datafold_name}_matches.json", "w"), indent=1)

# Repeat the same process for RLDATA10000
idx_train, idx_test = int(len(duplicated_ids_10000) * TRAIN), int(len(duplicated_ids_10000) * TEST)
train_ids, test_ids, val_ids = (
    duplicated_ids_10000[:idx_train],
    duplicated_ids_10000[idx_train : idx_train + idx_test],
    duplicated_ids_10000[idx_train + idx_test :],
)

idx_train, idx_test = int(len(entity_ids_10000) * TRAIN), int(len(entity_ids_10000) * TEST)
train_ids = np.concatenate((train_ids, entity_ids_10000[:idx_train]))
test_ids = np.concatenate((test_ids, entity_ids_10000[idx_train : idx_train + idx_test]))
val_ids = np.concatenate((val_ids, entity_ids_10000[idx_train + idx_test :]))

X_rldata10000["record_id"] = "id_C-" + X_rldata10000.index.astype(str)

X_rldata10000_train = X_rldata10000[X_rldata10000["entity_id"].isin(train_ids)]
X_rldata10000_test = X_rldata10000[X_rldata10000["entity_id"].isin(test_ids)]
X_rldata10000_val = X_rldata10000[X_rldata10000["entity_id"].isin(val_ids)]

X_rldata10000_train.to_csv(output_dir_10000 / "train-textual.csv", index=False)
X_rldata10000_test.to_csv(output_dir_10000 / "test-textual.csv", index=False)
X_rldata10000_val.to_csv(output_dir_10000 / "val-textual.csv", index=False)

# Save a json mapping each record_id with the record_ids with the same entity_id
datafold_names = ["train", "test", "val"]
datafolds = [X_rldata10000_train, X_rldata10000_test, X_rldata10000_val]
for datafold_name, datafold in zip(datafold_names, datafolds):
    rldata10000_datafold_matches = {}
    for record_id, entity_id in zip(datafold["record_id"], datafold["entity_id"]):
        rldata10000_datafold_matches[record_id] = datafold["record_id"][datafold["entity_id"] == entity_id].tolist()
        rldata10000_datafold_matches[record_id].remove(record_id)
    json.dump(rldata10000_datafold_matches, open(output_dir_10000 / f"{datafold_name}_matches.json", "w"), indent=1)
