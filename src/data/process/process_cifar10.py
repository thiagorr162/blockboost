import json
import pathlib
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

# Define input and output directory

INPUT_DIR = pathlib.Path("research/data/raw/cifar-10-batches-py/")
OUTPUT_DIR = pathlib.Path("research/data/processed/cifar10/")

# Make output directory if missing
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define unpickle function to unpickle the associated file


def unpickle(file):
    # Open given pickle file and unpickle it
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")

    # Unwrap dict structure, where 1st entrie is the label associated with each image and images is the actual image per se, in a vector of dimension 3072 (R + G + B)
    dict_values = [*dict.values()]
    labels = dict_values[1]
    images = dict_values[2]

    # Set structure to a pandas dataframe
    df_batch = pd.DataFrame(images)
    df_batch["entity_id"] = labels

    return df_batch


# Get each of the files decoded

for i in range(1, 6):
    batch_name = "data_batch_" + str(i)
    batch_df = unpickle(INPUT_DIR / batch_name)

    if i == 1:
        df_cifar = batch_df.copy()
    else:
        df_cifar = pd.concat([df_cifar, batch_df], ignore_index=True, axis=0)

df_cifar["record_id"] = df_cifar.index + 1
df_cifar.loc[:, ("entity_id")] = "id_C-" + df_cifar["entity_id"].astype(str)
df_cifar.loc[:, ("record_id")] = "id_C-" + df_cifar["record_id"].astype(str)

df_cifar.to_csv(OUTPUT_DIR / "train-vectorized.csv", index=False)


df_test = unpickle(INPUT_DIR / "test_batch")
df_test["record_id"] = df_test.index + 1 + 100000

df_test.loc[:, ("entity_id")] = "id_C-" + df_test["entity_id"].astype(str)
df_test.loc[:, ("record_id")] = "id_C-" + df_test["record_id"].astype(str)

df_test.to_csv(OUTPUT_DIR / "test-vectorized.csv", index=False)

for datafold_name, datafold in zip(["test", "train"], [df_test, df_cifar]):

    datafold_matches = {}

    for record_id, entity_id in tqdm(zip(datafold["record_id"], datafold["entity_id"])):
        datafold_matches[record_id] = datafold["record_id"][datafold["entity_id"] == entity_id].tolist()
        datafold_matches[record_id].remove(record_id)

    with open(OUTPUT_DIR / f"{datafold_name}_matches.json", "w") as f:
        json.dump(datafold_matches, f)
