import argparse
import json
import pathlib
import random
import shutil
import subprocess

import numpy as np
from tqdm import tqdm
from unique_names_generator import get_random_name


def get_git_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_shingles(row, k=2):
    concatenated_string = " ".join([str(a) for a in row])
    shingles = list(set([concatenated_string[i : i + k] for i in range(len(concatenated_string) - k + 1)]))
    return sorted(shingles)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--dataset",
    default="RLDATA500",
    help="Input dataset",
    type=str,
)

parser.add_argument(
    "-k",
    "--shingle_size",
    default=2,
    help="Size of shingles for LSH",
    type=int,
)
parser.add_argument(
    "-ci",
    "--column_indices",
    nargs="+",
    type=int,
    help="Indices of columns to be hashed.",
)

parser.add_argument(
    "-p",
    "--number_of_permutations",
    default=100,
    help="Number of permutations for creation of minhash vector",
    type=int,
)

parser.add_argument(
    "--seed",
    default=0,
    type=int,
)


parser.add_argument(
    "--shared",
    action="store_true",
    help="Save results in a shared folder instead of a personal one.",
)

args = parser.parse_args()

PROCESSED_DATA_PATH = pathlib.Path("research/data/processed")

name_seeed = vars(args).copy()
name_seeed["dataset"] = None  # database shouldn't affect name
name_seeed["shared"] = None  # using personal or shared folder shouldn't affect name
name_seeed = json.dumps(name_seeed)
random.seed(name_seeed)
unique_name = get_random_name(separator="_", style="lowercase")

INPUT_DIR = pathlib.Path(PROCESSED_DATA_PATH / args.dataset)
OUTPUT_DIR = pathlib.Path(PROCESSED_DATA_PATH / f"{args.dataset}-vectorize_minhash-{unique_name}-{args.shingle_size}")

if not args.shared:
    OUTPUT_DIR = "personal" / OUTPUT_DIR

STR_OUTPUT_DIR = str(OUTPUT_DIR)
print(f"{STR_OUTPUT_DIR=}")

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

shutil.copytree(INPUT_DIR, OUTPUT_DIR)

subsets = ["train", "test", "val"]


assert (
    not args.column_indices or min(args.column_indices) >= 0
), f"Error: out of bounds column index: {min(args.column_indices)}"


subset_to_number_of_entries = dict()

temporary_all_textual_path = OUTPUT_DIR / "all-textual.csv"
temporary_all_vectorized_path = OUTPUT_DIR / "all-vectorized.headless_csv"

# erase all-textual (if existent)
with (temporary_all_textual_path).open("w") as f:
    f.write("")

full_dataset = []

for subset in subsets:
    with (OUTPUT_DIR / f"{subset}-textual.csv").open("r") as f:
        all_lines = f.readlines()

    subset_to_number_of_entries[subset] = len(all_lines) - 1  # do not count header

    if subset == "train":
        lines = all_lines
    else:
        lines = all_lines[1:]

    for line in lines:
        line = line[:-1]
        cells = line.split(",")

        if args.column_indices:
            # make sure args.column_indices is valid
            assert max(args.column_indices) < len(
                cells
            ), f"Error: out of bounds column index: {max(args.column_indices)}"
            assert (
                len(cells) - 1 not in args.column_indices
            ), f"Error: forbidden column index: {len(cells)-1} (record_id)"
            assert (
                len(cells) - 2 not in args.column_indices
            ), f"Error: forbidden column index: {len(cells)-2} (entity_id)"
            all_column_indices = args.column_indices + [len(cells) - 2, len(cells) - 1]
            cells = [cells[k] for k in all_column_indices]

        with (temporary_all_textual_path).open("a") as f:
            full_dataset.append(cells)
            # f.write(",".join(cells) + "\n")

full_dataset = full_dataset[1:]  # remove header

dataset_without_ids = [row[:-2] for row in full_dataset]

shingled_records = [get_shingles(row, k=args.shingle_size) for row in dataset_without_ids]

# sort to make permutations deterministic
all_shingles = sorted(list(set([shingle for shingled_record in shingled_records for shingle in shingled_record])))

minhashed_records = np.zeros((len(dataset_without_ids), args.number_of_permutations), dtype=int)

np.random.seed(args.seed)
random.seed(args.seed)

for j in tqdm(range(args.number_of_permutations)):
    all_shingles_permut = np.random.permutation(all_shingles)
    shingle_to_index = {all_shingles_permut[i]: i for i in range(len(all_shingles_permut))}
    for i in range(len(dataset_without_ids)):
        minhashed_records[i, j] = min([shingle_to_index[s] for s in shingled_records[i]])

minhashed_dataset = [minhashed_records[i, :].tolist() + full_dataset[i][-2:] for i in range(len(full_dataset))]
minhashed_dataset = [[str(x) for x in entry] for entry in minhashed_dataset]


vectorized_lines = [",".join(x) + "\n" for x in minhashed_dataset]
header = ",".join([str(i) for i in range(args.number_of_permutations)] + ["entity_id", "record_id"]) + "\n"

with (OUTPUT_DIR / "train-vectorized.csv").open("w") as f:
    f.write(header)
    f.write("".join(vectorized_lines[: subset_to_number_of_entries["train"]]))

with (OUTPUT_DIR / "test-vectorized.csv").open("w") as f:
    f.write(header)
    f.write(
        "".join(
            vectorized_lines[
                subset_to_number_of_entries["train"] : subset_to_number_of_entries["train"]
                + subset_to_number_of_entries["test"]
            ]
        )
    )

with (OUTPUT_DIR / "val-vectorized.csv").open("w") as f:
    f.write(header)
    f.write("".join(vectorized_lines[subset_to_number_of_entries["train"] + subset_to_number_of_entries["test"] :]))
