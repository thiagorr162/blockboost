import argparse
import json
import pathlib
import pandas as pd
import shutil
import sys
from glob import glob

sys.path.insert(1, "src/utils")
import numpy as np
from tqdm import tqdm
from embedding_utils import get_SIFEmbedding, SIFEmbedding

parser = argparse.ArgumentParser()
parser.add_argument(
    "-db",
    "--database",
    default="RLDATA10000",
    help="Input dataset",
    type=str,
)

args = parser.parse_args()

PROCESSED_DATA_PATH = pathlib.Path("research/data/processed")

INPUT_DIR = pathlib.Path(PROCESSED_DATA_PATH / args.database)
OUTPUT_DIR = pathlib.Path(PROCESSED_DATA_PATH / f"{args.database}-fasttext")
OUTPUT_DIR.mkdir(exist_ok=True)

datafolds = ["train", "test", "val"]

database_metadata = json.load(open("src/data/process/datasets.json"))[args.database]

database_col_types = database_metadata["col_types"]

textual_columns = [col_name for col_name in database_col_types if database_col_types[col_name]['format'] == 'text']

non_textual_columns = [col_name for col_name in database_col_types if database_col_types[col_name]['format'] != 'text']


train_df = pd.read_csv(INPUT_DIR / "train-textual.csv")
test_df = pd.read_csv(INPUT_DIR / "test-textual.csv")
val_df = pd.read_csv(INPUT_DIR / "val-textual.csv")

train_df["_fold"] = "train"
test_df["_fold"] = "test"
val_df["_fold"] = "val"

data_df = pd.concat([train_df, test_df, val_df]).reset_index()

print(data_df)
data_df.fillna('', inplace=True)

tuple_embedding_data = get_SIFEmbedding(data_df.drop(['entity_id', 'record_id', '_fold'], axis=1), textual_columns)

print(tuple_embedding_data)

df_embedding = pd.DataFrame(tuple_embedding_data)

df_embedding = pd.concat([df_embedding, data_df], axis=1).drop(textual_columns+["index"], axis=1)

print(df_embedding)

if 'price' in list(df_embedding.columns):
    df_embedding['price'] = df_embedding['price'].str.replace('$', '')
    df_embedding['price'] = df_embedding['price'].str.replace('gbp', '') # computing the exchange rate might be better
    df_embedding['price'] = df_embedding['price'].replace('', np.nan)
    df_embedding['price'].astype(float)

emb = {}

emb["train"] = df_embedding[df_embedding['_fold'] == "train"].drop('_fold', axis=1)
emb["test"] = df_embedding[df_embedding['_fold'] == "test"].drop('_fold', axis=1)
emb["val"] = df_embedding[df_embedding['_fold'] == "val"].drop('_fold', axis=1)


for fold in ["train", "test", "val"]:
    print(emb[fold])

    for path in glob(f"{PROCESSED_DATA_PATH}/{args.database}/*"):
        shutil.copy(path, OUTPUT_DIR)

    emb[fold].to_csv( OUTPUT_DIR / f"{fold}-vectorized.csv", index=False)



#np.save(OUTPUT_DIR / "text_embedding", tuple_embedding_data)

print("Text data successfully vectorized.")





