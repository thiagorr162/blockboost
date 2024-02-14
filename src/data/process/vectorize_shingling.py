import argparse
import pathlib

import numpy as np
import pandas as pd


# Takes a string input and returns the set of k-shingles
# e.g. k=3: ["mathematics"] -> ["mat","ath","the","hem","ema","mat","ati","tic","ics"]
def build_shingles(sentence, k):
    shingles = []
    for i in range(len(sentence) - k + 1):
        shingles.append(sentence[i : i + k])
    return set(shingles)


# Converts list of k-shingle sets into a universal set of unique k-shinglesi
# Output is a dictionary that indexes unique shingles
def build_vocab(shingle_sets):
    universal_set = {item for set_ in shingle_sets for item in set_}
    vocab = {}
    for i, shingle in enumerate(list(universal_set)):
        vocab[shingle] = i
    return vocab


# transforms a set of k-shingles into the one-hot encoded vector
# vector has universal set size, and entry i is 1 if set contains i-th shingle, and 0 otherwise
def one_hot(shingles, vocab):
    vec = np.zeros(len(vocab))
    for shingle in shingles:
        if shingle in vocab:
            idx = vocab[shingle]
            vec[idx] = 1
    return vec


parser = argparse.ArgumentParser()
parser.add_argument(
    "-db",
    "--databases",
    default=["RLDATA500", "RLDATA10000"],
    help="Databases to vectorize by k-shingling.",
    type=list,
)
parser.add_argument(
    "--k",
    default=5,
    help="Number of nearest neighbors for LSH forest indexing",
    type=int,
)
parser.add_argument(
    "--str_column",
    "-sc",
    default="full_name",
    help="String column to shingle",
    type=str,
)

args = parser.parse_args()

PROCESSED_DATA_PATH = pathlib.Path("research/data/processed")
datasets = args.databases

for dataset in datasets:

    output_path = pathlib.Path(PROCESSED_DATA_PATH / dataset)
    datafold_names = ["train", "test", "val"]
    print(dataset)

    for data_fold in datafold_names:

        data = pd.read_csv(output_path / f"{data_fold}-textual.csv")

        # Get column of string data
        text_data = data[args.str_column]

        # build shingles
        shingles = []
        for name in text_data:
            shingles.append(build_shingles(name, args.k))

        # build shingle vocab if training data
        if data_fold == "train":
            vocab = build_vocab(shingles)

        # one-hot encode our shingles
        shingles_1hot = []
        for shingle_set in shingles:
            shingles_1hot.append(one_hot(shingle_set, vocab))

        # stack into single numpy array
        shingles_1hot = np.stack(shingles_1hot)

        df_shingled = pd.DataFrame(shingles_1hot)

        df_shingled["entity_id"] = data["entity_id"]
        df_shingled["record_id"] = data["record_id"]

        df_shingled.to_csv(output_path / f"{data_fold}-vectorized.csv", index=False)
