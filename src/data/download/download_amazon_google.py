import pathlib
from datetime import datetime

import pytz
import wget

OUTPUT_DIR = pathlib.Path("research/data/raw/amazon_google")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now(tz=pytz.timezone("America/Sao_Paulo")).isoformat("T", "milliseconds")
OUTPUT_PATH = OUTPUT_DIR / f"data-{timestamp}.zip"

wget.download(
    "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/amazon_google_raw_data.zip",
    str(OUTPUT_PATH),
)
