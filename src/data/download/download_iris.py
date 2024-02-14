import pathlib

import wget

OUTPUT_DIR = pathlib.Path("research/data/raw/iris")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

wget.download("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", str(OUTPUT_DIR / "iris.data"))
wget.download("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names", str(OUTPUT_DIR / "README"))
