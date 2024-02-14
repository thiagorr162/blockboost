import argparse
import json
import pathlib
import pandas as pd
import shutil
import sys
from glob import glob
import shutil

sys.path.insert(1, "src/utils")
import numpy as np
from tqdm import tqdm
from embedding_utils import get_SIFEmbedding, SIFEmbedding

parser = argparse.ArgumentParser()
parser.add_argument(
    "-v1",
    "--vectorization_1",
    default="RLDATA10000-fasttext",
    type=str,
)

parser.add_argument(
    "-v2",
    "--vectorization_2",
    default="RLDATA10000-vectorize_minhash-gray_bear-5",
    type=str,
)

args = parser.parse_args()

PROCESSED_DATA_PATH = pathlib.Path("research/data/processed")

INPUT_DIR_1 = pathlib.Path(PROCESSED_DATA_PATH / args.vectorization_1)
INPUT_DIR_2 = pathlib.Path(PROCESSED_DATA_PATH / args.vectorization_2)

database_name = args.vectorization_1.split("-")[0]
type_1 = "-".join(args.vectorization_1.split("-")[1:])
type_2 = "-".join(args.vectorization_2.split("-")[1:])


OUTPUT_DIR = pathlib.Path(PROCESSED_DATA_PATH / f"{database_name}-{type_1}-{type_2}")

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

shutil.copytree(INPUT_DIR_1, OUTPUT_DIR)
print(f"{OUTPUT_DIR=}")

for fold in ["train", "test", "val"]:
    df_1 = pd.read_csv(INPUT_DIR_1 / f"{fold}-vectorized.csv")
    df_1.drop("entity_id", inplace=True, axis=1)
    df_1.drop("record_id", inplace=True, axis=1)
    df_2 = pd.read_csv(INPUT_DIR_2 / f"{fold}-vectorized.csv")

    df_1 = df_1.add_suffix('-1')

    df_out = pd.concat([df_1, df_2], axis=1)
    #print(df_out)
    df_out.to_csv( OUTPUT_DIR / f"{fold}-vectorized.csv", index=False)

print("Text data successfully vectorized.")





