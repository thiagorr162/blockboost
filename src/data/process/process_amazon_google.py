import argparse
import json
import os
import pathlib
import shutil

import numpy as np
import pandas as pd

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

raw_dir = "research/data/raw/amazon_google/"
zip_file = raw_dir + "data-2022-10-04T15:15:53.186-03:00.zip"


np.random.seed(args.seed)

output_dir = pathlib.Path("research/data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_dir = output_dir / "amazon_google"
output_dir.mkdir(parents=True, exist_ok=True)
# First we unzip the download files
shutil.unpack_archive(zip_file, raw_dir)

# Then we read both datasets (tableA and tableB) and their list of matches
file_A = raw_dir + "tableA.csv"
file_B = raw_dir + "tableB.csv"
file_matches = raw_dir + "matches.csv"

df_A = pd.read_csv(file_A, encoding="unicode_escape")
df_B = pd.read_csv(file_B, encoding="unicode_escape")

df_A = df_A.replace(",", "&#44;", regex=True)
df_B = df_B.replace(",", "&#44;", regex=True)

# Column title in df_A and column name in df_B represent the same info
df_A = df_A.rename({"title": "name"}, axis="columns")

matches = pd.read_csv(file_matches, encoding="unicode_escape")

df_A["database"] = "id_A-"
df_B["database"] = "id_B-"

df_A["id"] = "id_A-" + df_A["id"].astype(str)
df_B["id"] = "id_B-" + df_B["id"].astype(str)

matches["idAmazon"] = "id_A-" + matches["idAmazon"].astype(str)
matches["idGoogleBase"] = "id_B-" + matches["idGoogleBase"].astype(str)

# assert ids from both datasets are unique
assert len(set(df_A["id"].values).intersection(set(df_B["id"].values))) == 0


idA_counted = matches["idAmazon"].value_counts()
duplicate_A_ids = idA_counted[idA_counted >= 2].index.values

idB_counted = matches["idGoogleBase"].value_counts()
duplicate_B_ids = idB_counted[idB_counted >= 2].index.values

df_A = df_A[~df_A["id"].isin(duplicate_A_ids)]
df_B = df_B[~df_B["id"].isin(duplicate_B_ids)]

matches = matches[~matches["idAmazon"].isin(duplicate_A_ids)]
matches = matches[~matches["idGoogleBase"].isin(duplicate_B_ids)]

df = pd.concat([df_A, df_B], ignore_index=True, axis=0)

matches_full = {}

for i in range(df.shape[0]):
    matches_full[str(i)] = []

for i in range(matches.shape[0]):

    id_A, id_B = matches.iloc[i]

    pandas_id_A = list(df[df["id"] == id_A].index)
    assert len(pandas_id_A) == 1

    pandas_id_B = list(df[df["id"] == id_B].index)
    assert len(pandas_id_B) == 1

    pandas_id_A = pandas_id_A[0]
    pandas_id_B = pandas_id_B[0]

    matches_full[str(pandas_id_A)].append(pandas_id_B)
    matches_full[str(pandas_id_B)].append(pandas_id_A)


id_to_entity_id = {}

A_ids = df[df["database"] == "id_A-"]["id"].values
B_ids = df[df["database"] == "id_B-"]["id"].values


for B_id in B_ids:

    B_match = matches[matches["idGoogleBase"] == B_id]

    if B_match.shape[0] == 1:
        id_to_entity_id[B_id] = B_match["idAmazon"].values[0]
    else:
        id_to_entity_id[B_id] = B_id

for A_id in A_ids:

    id_to_entity_id[A_id] = A_id

df["entity_id"] = df["id"].map(id_to_entity_id)
df["record_id"] = df["database"] + df.index.astype(str)

df = df.drop(columns=["id", "database"], axis=1)

df = df.fillna("")

dup_ids = df[df["entity_id"].duplicated()]["entity_id"].values
entity_ids = df[~df["entity_id"].duplicated(keep=False)]["entity_id"].values

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

df_train = df[df["entity_id"].isin(train_ids)]
df_test = df[df["entity_id"].isin(test_ids)]
df_val = df[df["entity_id"].isin(val_ids)]

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
