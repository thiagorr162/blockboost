#!/usr/bin/env bash

set -e

prediction_folder="$1"
eval_folder="$2"

# test if prediction_folder actually exists
[ -d "$prediction_folder" ]

mkdir -p "$eval_folder"

system_threads="$( grep -c ^processor /proc/cpuinfo )"

n_processes=1

ulimit -n `ulimit -Hn`

for datafold in "test" "train" "val"; do
    tsv_path="$prediction_folder/$datafold-hashes.tsv"
    tmp_eval_folder="$eval_folder/tmp_$datafold"

    [ -d "$tmp_eval_folder" ] && rm -r "$tmp_eval_folder"
    mkdir -p "$tmp_eval_folder"

    time ( seq 0 $(( n_processes - 1 )) )  | parallel "./bin/eval/hamming $n_processes {} < $tsv_path > $tmp_eval_folder/{}.txt"

    ( cat "$tmp_eval_folder"/*.txt | \
        awk -F'=' '/^tp/ {tp += $2} /^tn/ {tn += $2} /^fp/ {fp += $2} /^fn/ {fn += $2}  END {print(tp"\n"tn"\n"fp"\n"fn)}' ; \
        cat "$prediction_folder/hparams.json" | jq -c ) | \
        jq -s '{"recall": (.[0] / (.[0] + .[3])) , "precision": (.[0] / (.[0] + .[2])), "hparams": .[4] }'

done
