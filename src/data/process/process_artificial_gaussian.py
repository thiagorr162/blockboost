import argparse
import json
import pathlib

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--seed",
    default=42,
    help="Seed to random number generator.",
    type=int,
)
parser.add_argument(
    "-d",
    "--dimension",
    default=5,
    help="Number of feature dimensions.",
    type=int,
)
parser.add_argument(
    "-ne",
    "--number_of_entities",
    default=100,
    help="Number of entities.",
    type=int,
)
parser.add_argument(
    "-cv",
    "--covariance_matrix",
    help="Path to covariance matrix file as np.array.",
    type=str,
)
parser.add_argument(
    "-nr",
    "--number_of_records_per_entity",
    default=2,
    help="Number of records per entity.",
    type=int,
)
parser.add_argument(
    "-ss",
    "--split_seed",
    default=42,
    help="Split seed.",
    type=int,
)
parser.add_argument(
    "-sp",
    "--split_proportion",
    nargs=2,
    default=[0.70, 0.15],
    help="Split proportion of train and test used to generate train, test and val sets.",
    type=float,
)

args = parser.parse_args()

seed = args.seed
d = args.dimension
ne = args.number_of_entities
cv = pathlib.Path(args.covariance_matrix)
nr = args.number_of_records_per_entity

TRAIN, TEST = args.split_proportion[0], args.split_proportion[1]

raw_dir_artificial = pathlib.Path("research/data/raw/artificial_gaussian")
records_df = pd.read_csv(raw_dir_artificial / f"{cv.stem}-{ne}-{nr}.csv")

records_train = records_df[records_df["entity_id"].isin(range(int(ne * TRAIN)))]
records_test = records_df[records_df["entity_id"].isin(range(int(ne * TRAIN), int(ne * TRAIN) + int(ne * TEST)))]
records_val = records_df[records_df["entity_id"].isin(range(int(ne * TRAIN) + int(ne * TEST), ne))]

output_dir_artificial = pathlib.Path(f"research/data/processed/artificial_gaussian_{cv.stem}-{ne}-{nr}")
output_dir_artificial.mkdir(parents=True, exist_ok=True)

records_train.to_csv(
    output_dir_artificial / "train-vectorized.csv",
    index=False,
)
records_test.to_csv(
    output_dir_artificial / "test-vectorized.csv",
    index=False,
)
records_val.to_csv(
    output_dir_artificial / "val-vectorized.csv",
    index=False,
)

# Save a json mapping each record_id with the record_ids with the same entity_id
datafold_names = ["train", "test", "val"]
datafolds = [records_train, records_test, records_val]
for datafold_name, datafold in zip(datafold_names, datafolds):
    record_matches = {}
    for record_id, entity_id in zip(datafold["record_id"], datafold["entity_id"]):
        record_matches[int(record_id)] = datafold["record_id"][datafold["entity_id"] == entity_id].tolist()
        record_matches[int(record_id)].remove(int(record_id))
    json.dump(
        record_matches,
        open(
            output_dir_artificial / f"{datafold_name}_matches.json",
            "w",
        ),
        indent=1,
    )
