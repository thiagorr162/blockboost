import pathlib
import pickle

import pyreadr
import requests

raw_dir = pathlib.Path("research/data/raw/rldata")
raw_dir.mkdir(parents=True, exist_ok=True)

output_rldata10000 = raw_dir / "rldata10000.rda"
output_rldata500 = raw_dir / "rldata500.rda"

rldata10000_url = "https://github.com/cran/RecordLinkage/blob/master/data/RLdata10000.rda?raw=true"
rldata500_url = "https://github.com/cran/RecordLinkage/blob/master/data/RLdata500.rda?raw=true"

try:
    pyreadr.download_file(rldata10000_url, str(output_rldata10000))
    pyreadr.download_file(rldata500_url, str(output_rldata500))
except requests.exceptions.HTTPError as e:
    print(f"Error: {e}")
    print("Try Again!")
