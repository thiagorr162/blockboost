import argparse
import csv
import json
import pathlib

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--dataset_path",
    default="iris",
    help="Input dataset",
    type=str,
)

args = parser.parse_args()

PROCESSED_DATA_PATH = pathlib.Path("research/data/processed")

INPUT_DIR = pathlib.Path(args.dataset_path)

with (INPUT_DIR / "test-vectorized.csv").open("r") as f:
    dataset_lines = f.readlines()

# remove header
dataset_lines = dataset_lines[1:]
vectorization_matrix = np.zeros((len(dataset_lines), len(dataset_lines[0].split(",")) - 2), dtype=np.float64)

record_id_to_i = dict()
for i, line in enumerate(dataset_lines):
    record_id = line.split(",")[-1][:-1]
    print(type(record_id))
    record_id_to_i[record_id] = i

for i, line in enumerate(dataset_lines):
    row = [float(word) for word in line.split(",")[:-2]]
    vectorization_matrix[i, :] = row


distances = vectorization_matrix @ np.transpose(vectorization_matrix)
distances = np.sqrt(distances)

matches_mask = np.zeros(distances.shape, dtype=bool)

with (INPUT_DIR / "test_matches.json").open("r") as f:
    matches = json.load(f)

for k in matches:
    if k in record_id_to_i:
        i = record_id_to_i[k]
        for kk in matches[k]:
            kk = str(kk)
            if kk in record_id_to_i:
                j = record_id_to_i[kk]
                matches_mask[i, j] = 1

matches_distances = distances[matches_mask][:]
non_matches_distances = distances[0 == matches_mask][:]

for x in non_matches_distances:
    print(x)

print("&&")
for x in matches_distances:
    for k in range(int(len(non_matches_distances) / len(matches_distances))):
        print(x)
