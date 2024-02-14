import bisect
import pathlib

import numpy as np
import pandas as pd
from tqdm import tqdm

# Following function reads files in ivecs and fvecs format


def ivecs_read(fname):
    a = np.fromfile(fname, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view("float32")


# Setup io directories
INPUT_DIR = pathlib.Path("research/data/raw/sift/")
OUTPUT_DIR = pathlib.Path("research/data/processed/sift/")

# Make output directory if missing
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set seed parameter
seed = 1

# Turn on debug mode
debug_mode = False

# Read the fvec and ivec files
base = fvecs_read(INPUT_DIR / "sift_base.fvecs")
groundtruth = ivecs_read(INPUT_DIR / "sift_groundtruth.ivecs")
learn = fvecs_read(INPUT_DIR / "sift_learn.fvecs")
query = fvecs_read(INPUT_DIR / "sift_query.fvecs")

# Meaning of each matrix
# base: SIFT vectors of each of the 1M entries
# groundtruth: indices of the k = 100 nearest neighbors from Euclidean distance with respect to 10k query vectors
# learn: Training set for "quantizer"
# query: Vectors of query references

# Set base format
df_base = pd.DataFrame(base)
df_base["record_id"] = df_base.index + 1
df_base.to_csv(OUTPUT_DIR / "test_base.csv")

# Set query reference format
df_query = pd.DataFrame(query)
df_query["entity_id"] = df_query.index + 1
df_query.to_csv(OUTPUT_DIR / "test_query.csv")

# Form groundtruth pairs
groundtruth_pairs = []
idx = 1
for neighborhood in groundtruth:
    for neighbor in neighborhood:
        append_pair = [idx, neighbor]
        groundtruth_pairs.append(append_pair)

    idx += 1

# Write dataframe as csvd
df_groundtruth = pd.DataFrame(groundtruth_pairs)
df_groundtruth.to_csv(OUTPUT_DIR / "test_matches.csv", header=False, index=False)


# Separate training base and query datasets
df_learn = pd.DataFrame(learn)
df_learn["record_id"] = df_learn.index + 1
df_learn_query = df_learn.sample(frac=(df_query.shape[0] / base.shape[0]), random_state=seed)
df_learn_query.rename(columns={"record_id": "entity_id"})

# Save training base and query sets
df_learn_query.to_csv(OUTPUT_DIR / "train_query.csv")
df_learn.to_csv(OUTPUT_DIR / "train_base.csv")

# Determine training parameters
alpha = groundtruth.shape[1] / base.shape[0]
neighborhood_size = round(df_learn.shape[0] * alpha)


# Define bisected insert
def insert(seq, keys, item, max_len, keyfunc=lambda v: v):
    k = keyfunc(item)
    i = bisect.bisect_left(keys, k)
    if i < max_len:
        seq = np.insert(seq, i, item, axis=0)
        seq = seq[:-1, :]
    return seq


# Key function since flake is not a fan of function pointers
def first_elem(x):
    return x[0]


# Find truth for learn dataset

train_groundtruth_pairs = []
for idx_query, query_vec in tqdm(df_learn_query.iterrows()):
    dist_mat = np.ones([neighborhood_size, 2]) * np.inf
    query_vec = query_vec[:-1]
    for idx_base, base_vec in df_learn.iterrows():
        base_vec = base_vec[:-1]
        dist = np.linalg.norm(query_vec - base_vec)
        dist_mat = insert(dist_mat, dist_mat[:, 0], [dist, idx_base], neighborhood_size, first_elem)

    for idx_base in dist_mat[:, 1]:
        append_pair = [idx_query, idx_base]
        train_groundtruth_pairs.append(append_pair)

    if debug_mode:
        break


# Write the found pairs
df_train_groundtruth = pd.DataFrame(train_groundtruth_pairs)
df_train_groundtruth = df_train_groundtruth.astype(int)
df_train_groundtruth.to_csv(OUTPUT_DIR / "train_matches.csv", header=False, index=False)
