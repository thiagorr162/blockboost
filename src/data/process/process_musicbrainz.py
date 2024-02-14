"""
    This script is mostly a copy of lucasresenderc/bloss/blob/main/src/data/process/musicbrainz2M.py
    The data has the following entries:
    - TID: a unique record's id (in the complete dataset).
    - CID: cluster id (records having the same CID are duplicate)
    - CTID: a unique id within a cluster (if two records belong to the same cluster they will have the same CID but
    different CTIDs). These ids (CTID) start with 1 and grow until cluster size.
    - SourceID: identifies to which source a record belongs (there are five sources). The sources are deduplicated.
    - id: the original id from the source. Each source has its own Id-Format. Uniqueness is not guaranteed!!
    (can be ignored).
    - number: track or song number in the album.
    - length: the length of the track.
    - artist: the interpreter (artist or band) of the track.
    - year: date of publication.
    - language: language of the track.
"""

import argparse
import json
import pathlib
import re

import numpy as np
import pandas as pd


def clean_years(years):
    new_years = []
    for y in years:
        y = "".join(re.findall(r"\d+", str(y)))
        if y == "":
            y = 0
        else:
            y = int(y)
        new_years.append(y)
    return new_years


def clean_numbers(ns):
    new_ns = []
    for n in ns:
        n = int("0" + "".join(re.findall(r"\d+", str(n))))
        new_ns.append(n)
    return new_ns


def clean_lengths(lenghts):
    new_ls = []
    for lenght in lenghts:
        if lenght == "unk." or lenght == "unknown" or lenght == "n.a." or lenght == ".":
            lenght = "0"

        lenght = lenght.replace("m ", ":").replace("sec", "").replace("Sec", "").replace(" ", "")
        if ":" in lenght and lenght.replace(":", "").isnumeric():
            m, s = lenght.split(":")
            lenght = int(m) * 60 + int(s)
        elif "." in lenght and lenght.replace(".", "").isnumeric():
            lenght = int(float(lenght) * 60)
        elif len(lenght) > 4 and lenght.isnumeric():
            lenght = int(lenght) // 1000
        elif len(lenght) <= 4 and lenght.isnumeric():
            lenght = int(lenght)
        else:
            lenght = 0
        new_ls.append(lenght)
    return new_ls


def to_string(vs):
    new_vs = []
    for v in vs:
        new_vs.append(str(v))
    return new_vs


parser = argparse.ArgumentParser()
parser.add_argument(
    "-sp",
    "--split_proportion",
    nargs=2,
    default=[0.20, 0.60],
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

INPUT_DIR = pathlib.Path("research/data/raw/musicbrainz")

files_sulfix = ["20", "200", "2000"]

for sulfix in files_sulfix:
    print(f"Processing musicbrainz {sulfix}")

    OUTPUT_DIR = pathlib.Path(f"research/data/processed/musicbrainz_{sulfix}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = pd.read_csv(INPUT_DIR / f"musicbrainz_{sulfix}.csv")

    entries.fillna(
        {"number": 0, "title": "", "length": "0:0", "artist": "", "album": "", "year": 0, "language": ""}, inplace=True
    )
    entries["database"] = entries["SourceID"].replace({1: "id_A", 2: "id_B", 3: "id_C", 4: "id_D", 5: "id_E"})
    entries["entity_id"] = entries["CID"].astype(str)
    entries["record_id"] = entries["database"] + "-" + entries["TID"].astype(str)

    del entries["CTID"], entries["id"], entries["TID"], entries["CID"], entries["SourceID"], entries["database"]

    entries["year"] = clean_years(entries["year"])
    entries["number"] = clean_numbers(entries["number"])
    entries["length"] = clean_lengths(entries["length"])
    entries["title"] = to_string(entries["title"])
    entries["artist"] = to_string(entries["artist"])
    entries["album"] = to_string(entries["album"])
    entries["language"] = to_string(entries["language"])

    duplicated_ids = entries[entries["entity_id"].duplicated()]["entity_id"].unique()
    unique_ids = entries[~entries["entity_id"].duplicated(keep=False)]["entity_id"].unique()

    np.random.seed(args.seed)

    np.random.shuffle(duplicated_ids)
    np.random.shuffle(unique_ids)

    TRAIN, TEST = args.split_proportion[0], args.split_proportion[1]

    idx_train, idx_test = int(len(duplicated_ids) * TRAIN), int(len(duplicated_ids) * TEST)
    train_ids, test_ids, val_ids = (
        duplicated_ids[:idx_train],
        duplicated_ids[idx_train : idx_train + idx_test],
        duplicated_ids[idx_train + idx_test :],
    )

    idx_train, idx_test = int(len(unique_ids) * TRAIN), int(len(unique_ids) * TEST)
    train_ids = np.concatenate((train_ids, unique_ids[:idx_train]))
    test_ids = np.concatenate((test_ids, unique_ids[idx_train : idx_train + idx_test]))
    val_ids = np.concatenate((val_ids, unique_ids[idx_train + idx_test :]))

    entries_train = entries[entries["entity_id"].isin(train_ids)]
    entries_test = entries[entries["entity_id"].isin(test_ids)]
    entries_val = entries[entries["entity_id"].isin(val_ids)]

    entries_train.to_csv(OUTPUT_DIR / "train-textual.csv", index=False)
    entries_test.to_csv(OUTPUT_DIR / "test-textual.csv", index=False)
    entries_val.to_csv(OUTPUT_DIR / "val-textual.csv", index=False)

    datafold_names = ["train", "test", "val"]
    datafolds = [entries_train, entries_test, entries_val]
    for datafold_name, datafold in zip(datafold_names, datafolds):
        entries_datafold_matches = {}
        for record_id, entity_id in zip(datafold["record_id"], datafold["entity_id"]):
            entries_datafold_matches[record_id] = datafold["record_id"][datafold["entity_id"] == entity_id].tolist()
            entries_datafold_matches[record_id].remove(record_id)
        json.dump(entries_datafold_matches, open(OUTPUT_DIR / f"{datafold_name}_matches.json", "w"), indent=1)
