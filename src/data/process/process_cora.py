import json
import pathlib

import dgl
import torch
from dgl.data import CoraGraphDataset
from tqdm import tqdm

OUTPUT_DIR = pathlib.Path("research/data/processed/cora/")

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

dataset = CoraGraphDataset(raw_dir="research/data/raw/cora")


for subset_name in tqdm(["train", "test", "val"], "processing subsets"):
    csv_str = [",".join([str(a) for a in range(len(dataset[0].ndata["feat"][0]))]) + ",entity_id,record_id\n"]

    matches = {}

    entity_id_to_record_ids = {}

    mask_key = subset_name + "_mask"
    mask = dataset[0].ndata[mask_key]

    for i in dataset[0].nodes():
        if mask[i]:
            i = int(i)
            entity_id = "id_C-" + str(int(dataset[0].ndata["label"][i]))

            csv_str.append(
                ",".join([str(float(a)) for a in dataset[0].ndata["feat"][i]])
                + ","
                + entity_id
                + ","
                + "id_C-"
                + str(i)
                + "\n"
            )

            if entity_id not in entity_id_to_record_ids:
                entity_id_to_record_ids[entity_id] = []
            entity_id_to_record_ids[entity_id].append("id_C-" + str(i))

    csv_str = "".join(csv_str)

    with (OUTPUT_DIR / (subset_name + "-vectorized.csv")).open("w") as f:
        f.write(csv_str)

    for k in entity_id_to_record_ids:
        record_ids = entity_id_to_record_ids[k]
        for i in record_ids:
            for j in record_ids:
                if j != i:

                    if i not in matches:
                        matches[i] = []
                    matches[i].append(j)

    with (OUTPUT_DIR / (subset_name + "_matches.json")).open("w") as f:
        f.write(json.dumps(matches))
