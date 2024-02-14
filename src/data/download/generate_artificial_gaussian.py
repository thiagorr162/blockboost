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

args = parser.parse_args()

seed = args.seed
d = args.dimension
ne = args.number_of_entities
cv = pathlib.Path(args.covariance_matrix)
nr = args.number_of_records_per_entity

rng = np.random.default_rng(seed)

entities = rng.normal(loc=0, scale=1, size=(ne, d))
records = np.tile(entities, (nr, 1))

noise_covariance = np.load(args.covariance_matrix, "r")

assert (noise_covariance.shape[0] == noise_covariance.shape[1]) and noise_covariance.shape[0] == d

records[:, :] += rng.multivariate_normal(mean=np.zeros(d), cov=noise_covariance, size=nr * ne)

records_df = pd.DataFrame(records)
records_df["record_id"] = records_df.index
records_df["entity_id"] = np.tile(range(ne), nr)

output_dir_artificial = pathlib.Path("research/data/raw/artificial_gaussian")
output_dir_artificial.mkdir(parents=True, exist_ok=True)

records_df.to_csv(
    output_dir_artificial / f"{cv.stem}-{ne}-{nr}.csv",
    index=False,
)
