import pathlib
from datetime import datetime

import pytz
import wget

OUTPUT_DIR = pathlib.Path("research/data/raw/dblp_scholar")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now(tz=pytz.timezone("America/Sao_Paulo")).isoformat("T", "milliseconds")
OUTPUT_PATH = OUTPUT_DIR / f"data-{timestamp}.zip"

wget.download(
    "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-GoogleScholar/dblp_scholar_raw_data.zip",
    str(OUTPUT_PATH),
)
