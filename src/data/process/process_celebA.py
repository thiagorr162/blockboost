import json
import pathlib
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

INPUT_DIR = pathlib.Path("research/data/raw/celeba/")

split_list = pd.read_csv(INPUT_DIR / "list_eval_partition.txt", header=None, names=["file", "datafold"], delimiter=" ")
split_list = split_list.replace({0: "train", 1: "val", 2: "test"})

entity_list = pd.read_csv(INPUT_DIR / "identity_CelebA.txt", header=None, names=["file", "entity_id"], delimiter=" ")
attr_list = pd.read_csv(INPUT_DIR / "list_attr_celeba.txt", delimiter=" ")

metadata = pd.merge(pd.merge(split_list, entity_list), attr_list)
metadata["record_id"] = metadata.apply(lambda row: int(row["file"].replace(".jpg", "")), axis=1)

datafolds = ["train", "val", "test"]
for version in ["celebalanced", "celeba", "celebaby"]:
    for datafold in datafolds:
        OUTPUT_DIR = pathlib.Path(f"research/data/processed/{version}/")
        metadata["output_path"] = metadata.apply(
            lambda row: "research/data/processed/" + version + "/" + row["datafold"] + "/" + row["file"], axis=1
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / datafold).mkdir(parents=True, exist_ok=True)

        meta_filtered = metadata[metadata["datafold"] == datafold]
        matches = {}
        matches_with_path = {}

        entity_ids = np.unique(meta_filtered["entity_id"].values)

        # filter to create not full datasets
        if version == "celebaby":
            entity_ids = entity_ids[: int(len(entity_ids) / 100)]
            meta_filtered = meta_filtered[meta_filtered["entity_id"].isin(entity_ids)]

        if version == "celebalanced":
            keep = pd.Index([])
            for entity_id in entity_ids:
                keep = keep.union(meta_filtered.index[meta_filtered["entity_id"] == entity_id][:2])
            meta_filtered = meta_filtered[meta_filtered.index.isin(keep)]

        meta_filtered.to_csv(OUTPUT_DIR / f"{datafold}-metadata.csv", index=None)

        for entity_id in entity_ids:
            record_ids = meta_filtered["record_id"][meta_filtered["entity_id"] == entity_id].values.tolist()
            record_paths = meta_filtered["output_path"][meta_filtered["entity_id"] == entity_id].values.tolist()
            for i, record_id in enumerate(record_ids):
                matches[record_id] = record_ids[:i] + record_ids[i + 1 :]
            for i, record_path in enumerate(record_paths):
                matches_with_path[record_path] = record_paths[:i] + record_paths[i + 1 :]

        json.dump(matches, open(OUTPUT_DIR / f"{datafold}_matches.json", "w"))
        json.dump(matches_with_path, open(OUTPUT_DIR / f"{datafold}_matches_with_path.json", "w"))

        # copy images
        for filename in tqdm(meta_filtered["file"].values):
            shutil.copyfile(INPUT_DIR / f"imgs/img_align_celeba/{filename}", OUTPUT_DIR / datafold / filename)
