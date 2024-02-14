import json
import pathlib
from io import StringIO

import numpy as np

OUTPUT_DIR = pathlib.Path("research/data/processed/iris")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


raw_csv_path = pathlib.Path("research/data/raw/iris/iris.data")

out = "0,1,2,3,entity_id,record_id\n"

matrix = []
with raw_csv_path.open("r") as f:
    i = 0
    for line in f.readlines():
        if len(line) > 2:
            array_of_strings = line[:-1].split(",")
            out += (
                ",".join(array_of_strings[:-1]) + "," + "id_C-" + array_of_strings[-1] + "," + "id_C-" + str(i) + "\n"
            )
            matrix += [[float(x) for x in line[:-1].split(",Iris")[0].split(",")]]
            i += 1

matrix = np.array(matrix, dtype=float)
mult = matrix @ matrix.transpose()

matches = {"id_C-" + str(i): ["id_C-" + str(np.argmax(mult[i, :]))] for i in range(matrix.shape[0])}

str_matches = json.dumps(matches)

for subset in ["test", "val", "train"]:
    with (OUTPUT_DIR / (subset + "_matches.json")).open("w") as f:
        f.write(str_matches)
    with (OUTPUT_DIR / (subset + "-vectorized.csv")).open("w") as f:
        f.write(out)
