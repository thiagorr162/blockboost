import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    "-m_p",
    "--model_path",
    default="research/models/DeepLSH/celebalanced/-db=celebalanced-fdim=100-hdim=10-prb=0.6-hdist=l0-mc=8-ed=cosine-model=LeNet5-seed=0-ep=100-pt=5-lr=0.005-wd=0.0005-optim=sgd-clip=0.0",
    help="Path containing the model.",
    type=str,
)

parser.add_argument(
    "-ed", "--embedding_distance", default="cosine", help="Distance used in feature space for loss function.", type=str
)

args = parser.parse_args()

model_path = pathlib.Path(args.model_path)

eval_folder = pathlib.Path(str(model_path).replace("/models/", "/eval/")).parent
eval_path = eval_folder / model_path.name
eval_path.mkdir(parents=True, exist_ok=True)

# load dataloaders
data_folds = ["train", "val", "test"]

# Generate distance histogram
for data_fold in data_folds:

    distance_dictionary_path = model_path / f"{data_fold}-distance_distribution.json"

    pair_block_prob_dictionary_path = model_path / f"{data_fold}-pair_block_prob_distribution.json"

    # load distance dictionary
    with open(distance_dictionary_path, "r") as input_file:
        distance_dictionary = json.load(input_file)

    # get distances and their averages
    match_dist = distance_dictionary["match_dist"]
    non_match_dist = distance_dictionary["non_match_dist"]
    match_dist_emb = distance_dictionary["match_dist_emb"]
    non_match_dist_emb = distance_dictionary["non_match_dist_emb"]

    avg_match_dist = np.mean(match_dist)
    avg_non_match_dist = np.mean(non_match_dist)
    avg_match_dist_emb = np.mean(match_dist_emb)
    avg_non_match_dist_emb = np.mean(non_match_dist_emb)

    # load pair blocking probability dictionary and their averages
    with open(pair_block_prob_dictionary_path, "r") as input_file:
        pair_block_prob_dictionary = json.load(input_file)

    # get pair_block_probes
    match_pair_block_prob = pair_block_prob_dictionary["match_pair_block_prob"]
    non_match_pair_block_prob = pair_block_prob_dictionary["non_match_pair_block_prob"]
    match_pair_block_prob_emb = pair_block_prob_dictionary["match_pair_block_prob_emb"]
    non_match_pair_block_prob_emb = pair_block_prob_dictionary["non_match_pair_block_prob_emb"]

    avg_match_pair_block_prob = np.mean(match_pair_block_prob)
    avg_non_match_pair_block_prob = np.mean(non_match_pair_block_prob)
    avg_match_pair_block_prob_emb = np.mean(match_pair_block_prob_emb)
    avg_non_match_pair_block_prob_emb = np.mean(non_match_pair_block_prob_emb)

    fig, axs = plt.subplots(2, 1, figsize=(6, 10))
    plt.subplots_adjust(hspace=0.4)

    dist_min_orig = min(match_dist + non_match_dist)
    dist_max_orig = max(match_dist + non_match_dist)

    axs[0].set_title("Distance histogram in original feature space")
    axs[0].hist(
        match_dist,
        label="Match Distance",
        bins=25,
        color="cornflowerblue",
        density=True,
        range=(dist_min_orig, dist_max_orig),
    )
    axs[0].axvspan(avg_match_dist, avg_match_dist, color="blue")
    axs[0].hist(
        non_match_dist,
        label="Non Match Distance",
        bins=25,
        alpha=0.5,
        color="orange",
        density=True,
        range=(dist_min_orig, dist_max_orig),
    )
    axs[0].axvspan(avg_non_match_dist, avg_non_match_dist, color="orange")
    axs[0].legend(fontsize="small")
    axs[0].set_xlabel(f"{args.embedding_distance} distance")

    dist_min_emb = min(match_dist_emb + non_match_dist_emb)
    dist_max_emb = max(match_dist_emb + non_match_dist_emb)

    axs[1].set_title("Distance histogram in embedding space")
    axs[1].hist(
        match_dist_emb,
        label="Match Distance",
        bins=25,
        color="cornflowerblue",
        density=True,
        range=(dist_min_emb, dist_max_emb),
    )
    axs[1].axvspan(avg_match_dist_emb, avg_match_dist_emb, color="blue")
    axs[1].hist(
        non_match_dist_emb,
        label="Non Match Distance",
        bins=25,
        alpha=0.5,
        color="orange",
        density=True,
        range=(dist_min_emb, dist_max_emb),
    )
    axs[1].axvspan(avg_non_match_dist_emb, avg_non_match_dist_emb, color="orange")
    axs[1].legend(fontsize="small")
    axs[1].set_xlabel(f"{args.embedding_distance} distance")

    hist_out_path = eval_path / f"{data_fold}-distance_histogram.pdf"
    plt.savefig(hist_out_path)

    # Histogram for pair blocking probability terms

    fig, axs = plt.subplots(2, 1, figsize=(6, 10))
    plt.subplots_adjust(hspace=0.4)

    axs[0].set_title("PBP histogram in original feature space")
    axs[0].hist(match_pair_block_prob, label="Match PBP", bins=25, color="cornflowerblue", density=True, range=(0, 1))
    axs[0].axvspan(avg_match_pair_block_prob, avg_match_pair_block_prob, color="blue")
    axs[0].hist(
        non_match_pair_block_prob, label="Non Match PBP", bins=25, alpha=0.5, color="orange", density=True, range=(0, 1)
    )
    axs[0].axvspan(avg_non_match_pair_block_prob, avg_non_match_pair_block_prob, color="orange")
    axs[0].legend(fontsize="small")
    axs[0].set_xlabel(f"Pair blocking probability ({args.embedding_distance})")

    axs[1].set_title("PBP histogram in embedding space")
    axs[1].hist(
        match_pair_block_prob_emb, label="Match PBP", bins=25, color="cornflowerblue", density=True, range=(0, 1)
    )
    axs[1].axvspan(avg_match_pair_block_prob_emb, avg_match_pair_block_prob_emb, color="blue")
    axs[1].hist(
        non_match_pair_block_prob_emb,
        label="Non Match PBP",
        bins=25,
        alpha=0.5,
        color="orange",
        density=True,
        range=(0, 1),
    )
    axs[1].axvspan(avg_non_match_pair_block_prob_emb, avg_non_match_pair_block_prob_emb, color="orange")
    axs[1].legend(fontsize="small")
    axs[1].set_xlabel(f"Pair blocking probability ({args.embedding_distance})")

    hist_out_path = eval_path / f"{data_fold}-pair_block_prob_histogram.pdf"
    plt.savefig(hist_out_path)

    prediction_data_path = eval_path / f"{data_fold}-full_prediction.json"
    if prediction_data_path.exists():
        with open(prediction_data_path, "r") as input_file:
            prediction_dictionary = json.load(input_file)

        prediction_dictionary["average_pair_blocking_probability_matches"] = avg_match_pair_block_prob_emb
        prediction_dictionary["average_pair_blocking_probability_non_matches"] = avg_non_match_pair_block_prob_emb

        # store updated version of dictionary
        with open(prediction_data_path, "w") as output_file:
            output_file.write(json.dumps(prediction_dictionary, indent=4))
