import argparse
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
    default=5,
    help="Number of entities.",
    type=int,
)
parser.add_argument(
    "-nr",
    "--number_of_records_per_entity",
    default=2,
    help="Number of records per entity.",
    type=int,
)
parser.add_argument(
    "-cd",
    "--corrupted_dimensions",
    default=2,
    help="Number of corrupted dimensions.",
    type=int,
)
parser.add_argument(
    "-cl",
    "--corruption_location",
    default="deterministic",
    help="Location of the corruption (binomial or deterministic).",
    choices=["deterministic", "binomial"],
)
parser.add_argument(
    "-cs",
    "--corruption_scale",
    default=1.0,
    help="Intensity of corruption (i.e., deviation of the corruption).",
    type=float,
)

args = parser.parse_args()

seed = args.seed
d = args.dimension
ne = args.number_of_entities
nr = args.number_of_records_per_entity
cd = args.corrupted_dimensions
corruption_location = args.corruption_location
corruption_scale = args.corruption_scale

rng = np.random.default_rng(seed)

entities = rng.normal(loc=0, scale=1, size=(ne, d))
records = np.tile(entities, (nr, 1))

if corruption_location == "deterministic":
    records[:, :cd] += rng.normal(loc=0, scale=corruption_scale, size=(ne * nr, cd))
elif corruption_location == "binomial":
    for i in range(ne * nr):
        records[i, rng.choice(d, size=cd, replace=False)] += rng.normal(loc=0, scale=corruption_scale, size=cd)

records_df = pd.DataFrame(records)
records_df["record_id"] = records_df.index
records_df["entity_id"] = np.tile(range(ne), nr)

output_dir_artificial = pathlib.Path("research/data/raw/artificial")
output_dir_artificial.mkdir(parents=True, exist_ok=True)

records_df.to_csv(
    output_dir_artificial / f"s{seed}-d{d}-ne{ne}-nr{nr}-cd{cd}-{corruption_location}-{corruption_scale}.csv",
    index=False,
)
