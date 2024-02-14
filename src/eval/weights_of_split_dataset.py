import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d",
    "--dataset_path",
    default="research/data/processed/musicbrainz_20-combined-a0dc5acac3ee3ed0/test-vectorized.csv",
    type=str
)
parser.add_argument(
    "-f",
    "--features_path",
    default="research/models/blockboost-fp32/musicbrainz_20-combined-a0dc5acac3ee3ed0/s_0-pm_4-pn_16-n_5000/embedding/test_prediction.features",
    type=str,
)

args = parser.parse_args()

header = Path(args.dataset_path).open("r").readlines()[0].split(",")[:-2]

weights = json.load(Path(args.features_path).open("r"))

print(header)
print(weights)

field_2_w = {}

for (f, w) in weights:
    field = header[f].split("-")[1]
    print(f, w, field)

    if field not in field_2_w:
        field_2_w[field] = 0

    field_2_w[field] += w

field_2_w = {k: v for k, v in sorted(field_2_w.items(), key=lambda item: -item[1])}
print(json.dumps(field_2_w, indent=4))
