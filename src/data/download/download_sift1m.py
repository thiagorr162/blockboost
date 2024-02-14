# Download sift dataset from ftp link
# Following website can be used as reference:
# http://corpus-texmex.irisa.fr/
# For publications, use following papers:
# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
# https://hal.inria.fr/inria-00514462/en

import pathlib
import tarfile
import urllib.request

# Save URLs
SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
OUTPUT_DIR = pathlib.Path("research/data/raw/")

# Make output directory if missing
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Form request
ftpstream = urllib.request.urlopen(SIFT_URL)

# Extract the tarfile
tar = tarfile.open(fileobj=ftpstream, mode="r|gz")
tar.extractall(path=OUTPUT_DIR)
