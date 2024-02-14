import pathlib
from datetime import datetime

import pytz
import wget

OUTPUT_DIR = pathlib.Path("research/data/raw/abstract_titles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now(tz=pytz.timezone("America/Sao_Paulo")).isoformat("T", "milliseconds")

wget.download(
    "https://raw.githubusercontent.com/DrKevinOHare/Titles-Abstracts-dataset/main/abstracts.csv",
    str(OUTPUT_DIR / f"abstracts-{timestamp}.csv"),
)
wget.download(
    "https://raw.githubusercontent.com/DrKevinOHare/Titles-Abstracts-dataset/main/titles.csv",
    str(OUTPUT_DIR / f"titles-{timestamp}.csv"),
)
