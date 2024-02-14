import json
import pathlib
import re

import pandas as pd

experiments = {}

print("Getting DeepBlocker experiments...")

concatenated_file_path = pathlib.Path("research/eval/deepblocker/all_deepblocker.jsonl")

with open(concatenated_file_path, "r") as f:
    for line in f:
        output = json.loads(line)
        hash_name = output["output_path"]

        experiments[hash_name] = {}
        experiments[hash_name]["recall"] = float(output["recall"])
        experiments[hash_name]["reduction_ratio"] = float(output["rr"])
        experiments[hash_name]["database"] = output["dataset"]
        experiments[hash_name]["datafold"] = output["datafold"]
        experiments[hash_name]["model"] = output["embedding"]
        experiments[hash_name]["seed"] = 1.0


print("Getting LTH experiments...")

concatenated_file_path = pathlib.Path("research/eval/learn_to_hash/all_lth.jsonl")

with open(concatenated_file_path, "r") as f:
    for line in f:
        output = json.loads(line)
        hash_name = output["output_path"]

        experiments[hash_name] = {}
        experiments[hash_name]["recall"] = float(output["recall"])
        experiments[hash_name]["reduction_ratio"] = float(output["reduction_ratio"])
        experiments[hash_name]["database"] = output["hparams"]["database"].replace("_adahash", "")
        experiments[hash_name]["database"] = experiments[hash_name]["database"].split("-")[0]
        if "RLDATA" not in experiments[hash_name]["database"]:
            experiments[hash_name]["database"] = experiments[hash_name]["database"].lower()
        experiments[hash_name]["datafold"] = output["datafold"]
        experiments[hash_name]["model"] = output["hparams"]["model"]
        experiments[hash_name]["seed"] = 1.0

print("Getting KLSH experiments...")
concatenated_file_path = pathlib.Path("research/eval/klsh/all_klsh.jsonl")

with open(concatenated_file_path, "r") as f:
    for line in f:
        output = json.loads(line)

        hash_name = "-".join(
            [
                output["hparams"]["database"],
                str(output["hparams"]["number_of_random_projections"]),
                str(output["hparams"]["number_of_blocks"]),
                str(output["hparams"]["shingle_size"]),
                output["hparams"]["model"],
            ]
        )
        hash_name += f"/{output['subset']}-"

        if hash_name in experiments:
            experiments[hash_name]["recall"] += float(output["recall"])
            experiments[hash_name]["reduction_ratio"] += float(output["reduction_ratio"])
            experiments[hash_name]["seed"] += 1
        else:
            experiments[hash_name] = {}
            experiments[hash_name]["recall"] = float(output["recall"])
            experiments[hash_name]["reduction_ratio"] = float(output["reduction_ratio"])
            experiments[hash_name]["seed"] = 1.0
            experiments[hash_name]["database"] = output["hparams"]["database"]
            experiments[hash_name]["datafold"] = output["subset"]
            experiments[hash_name]["model"] = "klsh"

print("Getting TLSH experiments...")
concatenated_file_path = pathlib.Path("research/eval/tlsh/all_tlsh.jsonl")

with open(concatenated_file_path, "r") as f:
    for line in f:
        output = json.loads(line)

        hash_name = "-".join(
            [
                output["hparams"]["database"],
                str(output["hparams"]["number_of_buckets"]),
                str(output["hparams"]["shingle_size"]),
                output["hparams"]["model"],
            ]
        )
        hash_name += f"/{output['subset']}-"

        if hash_name in experiments:
            experiments[hash_name]["recall"] += float(output["recall"])
            experiments[hash_name]["reduction_ratio"] += float(output["reduction_ratio"])
            experiments[hash_name]["seed"] += 1
        else:
            experiments[hash_name] = {}
            experiments[hash_name]["recall"] = float(output["recall"])
            experiments[hash_name]["reduction_ratio"] = float(output["reduction_ratio"])
            experiments[hash_name]["seed"] = 1.0
            experiments[hash_name]["database"] = output["hparams"]["database"]
            experiments[hash_name]["datafold"] = output["subset"]
            experiments[hash_name]["model"] = "tlsh"

print("Getting Canopy experiments...")
concatenated_file_path = pathlib.Path("research/eval/canopy/all_canopy.jsonl")

with open(concatenated_file_path, "r") as f:
    for line in f:
        output = json.loads(line)
        hash_name = re.sub(r"/seed_[0-9]-", "/", output["output_path"])

        if hash_name in experiments:
            experiments[hash_name]["recall"] += float(output["recall"])
            experiments[hash_name]["reduction_ratio"] += float(output["reduction_ratio"])
            experiments[hash_name]["seed"] += 1
        else:
            experiments[hash_name] = {}
            experiments[hash_name]["recall"] = float(output["recall"])
            experiments[hash_name]["reduction_ratio"] = float(output["reduction_ratio"])
            experiments[hash_name]["seed"] = 1.0
            experiments[hash_name]["database"] = output["hparams"]["database"]
            experiments[hash_name]["datafold"] = output["subset"]
            experiments[hash_name]["model"] = "canopy"

print("Getting BlockBoost experiments...")
concatenated_file_path = pathlib.Path("research/eval/blockboost/all_emb.jsonl")

with open(concatenated_file_path, "r") as f:
    for line in f:
        output = json.loads(line)
        hash_name = re.sub(r"/s_[0-9]-", "/", output["output_path"])

        if hash_name in experiments:
            experiments[hash_name]["recall"] += float(output["recall"])
            experiments[hash_name]["reduction_ratio"] += float(output["reduction_ratio"])
            experiments[hash_name]["seed"] += 1
        else:
            experiments[hash_name] = {}
            experiments[hash_name]["recall"] = float(output["recall"])
            experiments[hash_name]["reduction_ratio"] = float(output["reduction_ratio"])
            experiments[hash_name]["seed"] = 1.0
            experiments[hash_name]["database"] = output["hparams"]["database"].split("-")[0]
            experiments[hash_name]["datafold"] = output["datafold"].split("-")[0]
            experiments[hash_name]["model"] = "blockboost"

print("Getting BlockBoost (fp32) experiments...")
concatenated_file_path = pathlib.Path("research/eval/blockboost-fp32/all_emb.jsonl")

with open(concatenated_file_path, "r") as f:
    for line in f:
        output = json.loads(line)
        hash_name = re.sub(r"/s_[0-9]-", "/", output["output_path"])

        if hash_name in experiments:
            experiments[hash_name]["recall"] += float(output["recall"])
            experiments[hash_name]["reduction_ratio"] += float(output["reduction_ratio"])
            experiments[hash_name]["seed"] += 1
        else:
            experiments[hash_name] = {}
            experiments[hash_name]["recall"] = float(output["recall"])
            experiments[hash_name]["reduction_ratio"] = float(output["reduction_ratio"])
            experiments[hash_name]["seed"] = 1.0
            experiments[hash_name]["database"] = output["hparams"]["database"].split("-")[0]
            experiments[hash_name]["datafold"] = output["datafold"].split("-")[0]
            experiments[hash_name]["model"] = "blockboost-fp32"

for hash_name in experiments:
    experiments[hash_name]["recall"] = experiments[hash_name]["recall"] / experiments[hash_name]["seed"]
    experiments[hash_name]["reduction_ratio"] = (
        experiments[hash_name]["reduction_ratio"] / experiments[hash_name]["seed"]
    )

    experiments[hash_name]["rrr"] = (
        2
        * experiments[hash_name]["recall"]
        * experiments[hash_name]["reduction_ratio"]
        / (experiments[hash_name]["recall"] + experiments[hash_name]["reduction_ratio"])
    )

print("Creating final dataset...")

df = pd.DataFrame(experiments).T

output_val_path = pathlib.Path("research/paper-blockboost/tables/val/")
output_val_path.mkdir(parents=True, exist_ok=True)

output_test_path = pathlib.Path("research/paper-blockboost/tables/test/")
output_test_path.mkdir(parents=True, exist_ok=True)

print("Saving...")
for db in df["database"].unique():
    if db == "musicbrainz_200":
        continue
    best_val_models = df.loc[(df["database"] == db) * (df["datafold"] == "val")]
    best_val_models = best_val_models.sort_values(by="rrr", ascending=False).drop_duplicates("model")

    best_val_models.to_csv(output_val_path / f"{db}.csv")

    best_val_indices = []
    for idx in best_val_models.index:
        best_val_indices.append(idx.replace("/val-", "/test-"))

        best_test_models = df.loc[(df["database"] == db) * (df["datafold"] == "test")]
        best_test_models = best_test_models.loc[best_val_indices].sort_values(by="rrr", ascending=False)

    best_test_models.to_csv(output_test_path / f"{db}.csv")
