import argparse
import json
from pathlib import Path
import pandas as pd
import shutil
import sys
from glob import glob
import shutil
import hashlib

sys.path.insert(1, "src/utils")
import numpy as np
from tqdm import tqdm
from embedding_utils import get_SIFEmbedding, SIFEmbedding

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--dataset_paths",
    default=[
        "research/data/processed/musicbrainz_20-split-album-vectorize_minhash-sapphire_jay-4",
        "research/data/processed/musicbrainz_20-split-artist-vectorize_minhash-sapphire_jay-4",
        "research/data/processed/musicbrainz_20-split-language-vectorize_minhash-sapphire_jay-4",
        "research/data/processed/musicbrainz_20-split-length-vectorize_minhash-sapphire_jay-4",
        "research/data/processed/musicbrainz_20-split-number-vectorize_minhash-sapphire_jay-4",
        "research/data/processed/musicbrainz_20-split-title-vectorize_minhash-sapphire_jay-4",
        "research/data/processed/musicbrainz_20-split-year-vectorize_minhash-sapphire_jay-4",
    ],
    type=str,
    nargs="+"
)


args = parser.parse_args()

PROCESSED_DATA_PATH = Path("research/data/processed")

INPUT_DIRS = [Path(p) for p in args.dataset_paths]

h = hashlib.new('sha256')
h.update(','.join([str(p) for p in INPUT_DIRS[:]]).encode())
hashcode = h.hexdigest()[:16]

OUTPUT_DIR = Path(str(INPUT_DIRS[0]).split("-")[0] + f"-combined-{hashcode}")

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

shutil.copytree(INPUT_DIRS[0], OUTPUT_DIR)

for fold in ["train", "test", "val"]:
    df_list = []

    for INPUT_DIR in INPUT_DIRS[:-1]:
        df = pd.read_csv(INPUT_DIR / f"{fold}-vectorized.csv")
        df.drop("entity_id", inplace=True, axis=1)
        df.drop("record_id", inplace=True, axis=1)
        df = df.add_suffix("-" + INPUT_DIR.name.split("-")[2])
        df_list.append(df)

    df_final = pd.read_csv(INPUT_DIRS[-1] / f"{fold}-vectorized.csv")
    df_final = df_final.rename(columns={c: c+"-"+INPUT_DIRS[-1].name.split("-")[2] for c in df_final.columns if c not in ['entity_id', 'record_id']})
    df_list.append(df_final)

    df_out = pd.concat(df_list, axis=1)
    #print(df_out)
    df_out.to_csv( OUTPUT_DIR / f"{fold}-vectorized.csv", index=False)

with Path(f"{OUTPUT_DIR}/hparams.json").open("w") as f:
    f.write(json.dumps(vars(args), indent=4)+"\n")

print("Text data successfully vectorized.")
print(f"{OUTPUT_DIR=}")


