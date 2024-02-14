import csv
import json
from glob import glob

import numpy as np
from tabulate import tabulate

per_dataset_tables = glob(
    "research/experiments_adahash/tables/best_models/recall_rr/*-best_experiment_given_model.json"
)

model_and_dataset_to_rrr = {}
model_and_dataset_to_rr = {}
model_and_dataset_to_recall = {}
model_and_dataset_to_max_rrr_val = {}

translation = {
    # models
    "adahash-random_features-full": "BlockBooster",
    "adahash-random_features-transitive": "BlockBooster (transitive)",
    "adahash-standard-full": "LSH",
    "adahash-standard-transitive": "LSH (transitive)",
    "RLDATA500": "rldata500",
    "RLDATA10000": "rldata10k",
    "dblp_scholar": "dblp_sch",
    # datasets
    "amazon_google": "amz_gg",
    "walmart_amazon": "wm_amz",
}


def format_result(value, bold=False):
    if bold:
        return "$\\mathbf{" + f"{value:.3f}" + "}$"
    else:
        return "$" + f"{value:.3f}" + "$"


for path in per_dataset_tables:
    with open(path, "r") as f:
        best_experiments = json.load(f)

    for best_model in best_experiments:
        dataset = "-".join(path.split("/")[-1].split("-")[:-1])
        if type(best_model["output_path"]) == str:
            best_model["output_path"] = [best_model["output_path"]]

        model = best_model["output_path"][0].split("/")[2]

        if (
            dataset not in ["cora", "cora_full", "cifar10"]
            and (not model.startswith("adahash-random_projection"))
            and ("transitive" not in model)
        ):

            # dataset = model["dataset"]
            # model = model["model"]
            # print()
            # print()
            # print(f"{dataset=}")
            # print(f"{model=}")
            # print(f"{best_model=}")

            if model in translation:
                model = translation[model]
            if dataset in translation:
                dataset = translation[dataset]

            rrr_vec = []
            recall_vec = []
            rr_vec = []
            for test_path in best_model["output_path"]:
                val_path = test_path.replace("/test-", "/val-")
                with open(val_path, "r") as f:
                    val_data = json.load(f)

                rr_vec.append(val_data["reduction_ratio"])
                recall_vec.append(val_data["recall"])

                if val_data["recall"] == 0 and val_data["reduction_ratio"] == 0:
                    rrr_vec.append(0)
                else:
                    rrr_vec.append(
                        2
                        * val_data["recall"]
                        * val_data["reduction_ratio"]
                        / (val_data["recall"] + val_data["reduction_ratio"])
                    )

            rrr = np.mean(rrr_vec)
            rr = np.mean(rr_vec)
            recall = np.mean(recall_vec)

            rrr_val = best_model["rrr"]

            # print(f"{rrr_vec=}")

            if ((model, dataset) not in model_and_dataset_to_max_rrr_val) or (
                model_and_dataset_to_max_rrr_val[(model, dataset)] < rrr_val
            ):
                model_and_dataset_to_rrr[(model, dataset)] = rrr  # bold( f"${rrr:.3f}$" )
                model_and_dataset_to_rr[(model, dataset)] = rr  # bold( f"${rrr:.3f}$" )
                model_and_dataset_to_recall[(model, dataset)] = recall  # bold( f"${rrr:.3f}$" )

                model_and_dataset_to_max_rrr_val[(model, dataset)] = rrr_val

        # for row in list(reader)[1:]:
        #    dataset = '-'.join(path.split('/')[-1].split('-')[:-1])
        #    model = row[0]
        #    rrr = float(row[1])

        #    if model in translation:
        #        model = translation[model]
        #    if dataset in translation:
        #        dataset = translation[dataset]

        #    if dataset not in ['cora', 'cora_full', 'cifar10']:
        #        if not model.startswith("adahash-random_projection"):
        #            if "transitive" not in model:
        #                model_and_dataset_to_rrr[(model,dataset)] =  rrr # bold( f"${rrr:.3f}$" )

# exit(0)

model_and_dataset_to_rrr = model_and_dataset_to_rrr


models = sorted(list({k[0] for k in model_and_dataset_to_rrr}))
datasets = sorted(list({k[1] for k in model_and_dataset_to_rrr}))

for model in models:
    avg = np.mean(
        [float(model_and_dataset_to_rrr[(model, d)]) for d in datasets if (model, d) in model_and_dataset_to_rrr]
    )
    model_and_dataset_to_rrr[(model, "avg")] = avg

datasets.append("avg")

for d in datasets:
    list_of_model_results = [
        [model_and_dataset_to_rrr[(m, d)], m] if (m, d) in model_and_dataset_to_rrr else [-1, m] for m in models
    ]
    list_of_model_results.sort(reverse=True)

    n_best_models = 3
    for i, r in enumerate(list_of_model_results):
        m = r[1]
        unformated_result = model_and_dataset_to_rrr[(m, d)] if (m, d) in model_and_dataset_to_rrr else -1
        model_and_dataset_to_rrr[(m, d)] = format_result(unformated_result, bold=i < 2)


rows = [[""] + ["" + d.replace("_", "\\_") + "" for d in datasets]]

for model in models:
    list_of_rrr = [
        model_and_dataset_to_rrr[(model, d)] if (model, d) in model_and_dataset_to_rrr else " . " for d in datasets
    ]
    rows.append([model] + list_of_rrr)


print(tabulate(rows, headers="firstrow", tablefmt="latex_raw"))

# print(','+','.join(datasets))
# for model in models:
#    print( model + ',' + ','.join([model_and_dataset_to_rrr[(model, d)] if (model,d) in model_and_dataset_to_rrr else ' . ' for d in datasets]))
