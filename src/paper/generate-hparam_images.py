import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

with open("research/eval/adahash/all_tests.jsonl", "r") as f:
    lines_jsonl = f.readlines()


# {'database': 'dblp_scholar-vectorize_minhash-gray_bear-5', 'data_type': 'vectorized', 'proportion_matches': 1, 'proportion_nonmatches': 1, 'n_iter': 5000, 'nonmatches_type': 'random', 'seed': 1, 'predict_train': False, 'bucket_size': 20, 'n_buckets': 15, 'lsh_type': 'standard', 'full_prediction': True, 'minimum_hamming_distance': 0, 'skip_existing': True, 'model': 'adahash', 'commit': '0e6c97c9a15d542df1d511b21cec993062e8fbe1'}, 'output_path': 'research/eval/adahash/dblp_scholar-vectorize_minhash-gray_bear-5/s_1-pm_1.0-pn_1.0-n_5000-nt_random/bs_20-nb_15-lt_standard/test-full_prediction.json', 'prediction_type': 'full', 'subset': 'test', 'evaluation_time': 0.24607110023498535}

OUTPUT_DIR = "research/paper/hparam_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

hparam_selection = {
    "n_iter": {5000},
    "lsh_type": {"random_features"},
    "proportion_nonmatches": {2},
}


hparam_range = {}

database_bs_nb_seed_to_rrr = {}

for line in tqdm(lines_jsonl):
    evaluation = json.loads(line)

    if evaluation["prediction_type"] != "full":
        continue

    satisfy_selection = [
        evaluation["hparams"][key] in hparam_selection[key] for key in hparam_selection if key in evaluation["hparams"]
    ]
    if False in satisfy_selection:
        continue

    for key in evaluation["hparams"]:
        if key not in hparam_range:
            hparam_range[key] = set()
        hparam_range[key].add(evaluation["hparams"][key])

    db = evaluation["hparams"]["database"]
    bs = evaluation["hparams"]["bucket_size"]
    nb = evaluation["hparams"]["n_buckets"]
    seed = evaluation["hparams"]["seed"]

    assert (db, bs, nb, seed) not in database_bs_nb_seed_to_rrr

    rrr = (
        2
        * evaluation["recall"]
        * evaluation["reduction_ratio"]
        / (evaluation["reduction_ratio"] + evaluation["recall"])
    )
    database_bs_nb_seed_to_rrr[(db, bs, nb, seed)] = rrr

for database in tqdm(hparam_range["database"], desc="saving plots"):
    xticks = sorted(list(hparam_range["n_buckets"]))
    yticks = sorted(list(hparam_range["bucket_size"]), reverse=True)
    bs_nb_to_avg_rrr = np.array(
        [
            [
                np.mean([database_bs_nb_seed_to_rrr[(database, bs, nb, seed)] for seed in hparam_range["seed"]])
                for nb in xticks
            ]
            for bs in yticks
        ]
    )

    # print(bs_nb_to_avg_rrr.shape)
    # image = Image.fromarray(bs_nb_to_avg_rrr)
    # image.save('test.png')
    # print(hparam_range)

    # library
    # Default heatmap: just a visualization of this square matrix
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    # import numpy as np
    ## Create a dataset
    # df = pd.DataFrame(, columns=["a","b","c","d","e"])

    sns.heatmap(bs_nb_to_avg_rrr)
    # default_x_ticks = range(len(x))
    # plt.plot(default_x_ticks, y)
    # plt.xticks(default_x_ticks, x)
    plt.xticks(np.array(range(len(xticks))) + 0.5, xticks)
    plt.yticks(np.array(range(len(yticks))) + 0.5, yticks)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.xlabel("L")
    plt.ylabel("k")
    # plt.title(database)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{database}.pdf")
    plt.close()
