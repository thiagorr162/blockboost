import numpy as np
import subprocess
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

from tqdm import tqdm

def run(command):
    out, err = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    if out:
        out = out.decode('utf-8')
        out = out.strip()
    else:
        out = None

    if err:
        err = err.decode('utf-8')
        err = err.strip()
    else:
        err = None

    return out, err

datasets = [
    "abt_buy",
    "amazon_google",
    "dblp_acm",
    "dblp_scholar",
    #"musicbrainz_20",
    "restaurant",
    "RLDATA10000",
    "walmart_amazon",
]

OUTPUT_FOLDER = Path("research/eval/similarity")
OUTPUT_FOLDER.mkdir(exist_ok=True)

MATRIX_PATH = OUTPUT_FOLDER /"m-ctt.npy"

m = np.zeros((len(datasets), len(datasets)), dtype=float)

if not MATRIX_PATH.exists():
    for ia, da in enumerate(tqdm(datasets, desc='computing similarities')):
        for ib, db in enumerate(datasets):
            out, err = run((
                f"bin/data/eval/similarity_measure "
                f"research/data/processed/{da}-ctt_3/test-vectorized.csv "
                f"research/data/processed/{db}-ctt_3/val-vectorized.csv "
                 "300"
            ))
            m[ia, ib] = float(out)

            print(out)
    np.save(MATRIX_PATH, m)
else:
   m = np.load(MATRIX_PATH)

print(m)


fig = plt.figure(figsize=(10, 10))
ax = sns.heatmap(m, annot=True, cmap='Blues', square=True)

ax.set_xlabel("Test", fontsize=14, labelpad=20)
ax.xaxis.set_ticklabels(datasets, rotation = 45)

ax.set_ylabel("Validation", fontsize=14, labelpad=20)
ax.yaxis.set_ticklabels(datasets, rotation = 45)

plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "similarity_table-ctt.pdf")
