# download music brainz 2M from https://www.informatik.uni-leipzig.de/~saeedi/musicbrainz-2000-A01.csv.dapo

import pathlib

import wget

OUTPUT_DIR = pathlib.Path("research/data/raw/musicbrainz")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files_to_download = ["20", "200", "2000"]

for file in files_to_download:

    print(f"\nDownloading musicbrainz {file}")
    OUTPUT_PATH = OUTPUT_DIR / f"musicbrainz_{file}.csv"
    wget.download(
        f"https://www.informatik.uni-leipzig.de/~saeedi/musicbrainz-{file}-A01.csv.dapo",
        str(OUTPUT_PATH),
    )
