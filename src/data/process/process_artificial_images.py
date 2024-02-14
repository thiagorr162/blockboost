import argparse
import json
import pathlib

import matplotlib.image as img
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--seed",
    default=42,
    help="Seed to random number generator.",
    type=int,
)
parser.add_argument(
    "-d",
    "--dimensions",
    nargs=2,
    default=[218, 178],
    help="Image dimension.",
    type=int,
)
parser.add_argument(
    "-p",
    "--patterns",
    default="balanced-8",
    help="Name of .json patterns in raw dir.",
    type=str,
)
parser.add_argument(
    "-nr",
    "--number_of_records_per_entity",
    default=2,
    help="Number of records per entity.",
    type=int,
)
parser.add_argument(
    "-sp",
    "--split_proportion",
    nargs=2,
    default=[0.70, 0.15],
    help="Split proportion of train and test used to generate train, test and val sets.",
    type=float,
)
parser.add_argument(
    "-name",
    "--database_name",
    default="artificial_image",
    help="Database name.",
    type=str,
)

args = parser.parse_args()

# get args
rng = np.random.default_rng(args.seed)
w, h = args.dimensions
train_prop, test_prop = args.split_proportion

# create dirs
PROCESSED_DIR = pathlib.Path("research/data/processed") / args.database_name
(PROCESSED_DIR / "train").mkdir(parents=True, exist_ok=True)
(PROCESSED_DIR / "test").mkdir(parents=True, exist_ok=True)
(PROCESSED_DIR / "val").mkdir(parents=True, exist_ok=True)

# create patterns
patterns = json.load(open(pathlib.Path("research/data/raw/artificial_images") / f"{args.patterns}.json"))

# computes the number of entities
n_entities = 1
for pattern in patterns:
    n_entities *= pattern["n_colors"]
n_imgs = n_entities * args.number_of_records_per_entity

print("Generating images...")
filenames = []
datafolds = []
entity_ids = []
record_ids = []
output_paths = []
for entity_id in tqdm(range(n_entities)):
    if entity_id < train_prop * n_entities:
        datafold = "train"
    elif entity_id < (train_prop + test_prop) * n_entities:
        datafold = "test"
    else:
        datafold = "val"

    for i in range(args.number_of_records_per_entity):
        # generate image
        image = np.zeros(shape=(w, h, 3))
        for pattern in patterns:
            x1, y1 = pattern["start"]
            x2, y2 = pattern["end"]
            color = pattern["colors"][entity_id % pattern["n_colors"]]
            noise_scale = pattern["noise_scale"]
            if type(color) == list:
                image[x1:x2, y1:y2, :] = rng.normal(loc=0, scale=noise_scale, size=(x2 - x1, y2 - y1, 3))
                image[x1:x2, y1:y2, 0] += color[0]
                image[x1:x2, y1:y2, 1] += color[1]
                image[x1:x2, y1:y2, 2] += color[2]
            else:
                image[x1:x2, y1:y2, :] = rng.normal(loc=color, scale=noise_scale, size=(x2 - x1, y2 - y1))[:, :, None]
        image[image < 0] = 0
        image[image > 255] = 255
        image /= 255

        # save image
        img_count = n_entities * i + entity_id
        filename = str(img_count).zfill(int(np.log10(n_imgs)) + 1) + ".jpg"
        img.imsave(PROCESSED_DIR / datafold / filename, image)

        # store metadata
        filenames.append(filename)
        datafolds.append(datafold)
        entity_ids.append(entity_id)
        record_ids.append(img_count)
        output_paths.append(str(PROCESSED_DIR / datafold / filename))

df = pd.DataFrame(
    {
        "file": filenames,
        "datafold": datafolds,
        "entity_id": entity_ids,
        "record_id": record_ids,
        "output_path": output_paths,
    }
)

for datafold in ["train", "test", "val"]:
    (df[df["datafold"] == datafold]).to_csv(PROCESSED_DIR / f"{datafold}-metadata.csv", index=False)

# Save a json mapping each record_id with the record_ids with the same entity_id
datafold_names = ["train", "test", "val"]
for datafold_name in datafold_names:
    datafold = df[df["datafold"] == datafold_name]
    record_matches = {}
    for record_id, entity_id in zip(datafold["record_id"], datafold["entity_id"]):
        record_matches[int(record_id)] = datafold["record_id"][datafold["entity_id"] == entity_id].tolist()
        record_matches[int(record_id)].remove(int(record_id))
    json.dump(
        record_matches,
        open(
            PROCESSED_DIR / f"{datafold_name}_matches.json",
            "w",
        ),
        indent=1,
    )
