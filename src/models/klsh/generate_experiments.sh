#!/bin/bash

source src/utils/trap_errors.sh

databases=(
    "RLDATA10000"
    "RLDATA500"
    "abt_buy"
    "amazon_google"
    "dblp_acm"
    "dblp_scholar"
    "restaurant"
    "walmart_amazon"
    #"cora"
    #"cora_full"
    #"iris"
    #"cifar10"
    "musicbrainz_20"
)

for d in "${list_of_datasets[@]}" ; do
    echo $d
done

seeds=( $( seq 0 3 ) )

n_blocks=(
    20
    40
    60
    80
    100
    250
    500
    1000
    1500
)

shingle_size=( $( seq 1 6 ) )

mkdir -p out/klsh/predict
echo "Running predictions (outputs in out/klsh/predict)..."

start_time=$SECONDS
parallel --halt-on-error 2  --progress \
    'python src/models/klsh/predict.py -d {1} --number_of_blocks {2} --shingle_size {3} --seed {4} &> out/klsh/predict/{#}.out' \
    ::: "${databases[@]}" \
    ::: "${n_blocks[@]}" \
    ::: "${shingle_size[@]}" \
    ::: "${seeds[@]}"
elapsed=$(( SECONDS - start_time ))
echo
eval "echo Total prediction time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
echo
echo  "Evaluating runs"
./src/eval/eval.sh klsh

echo
echo "Concatenating jsons of the test subset into 'research/eval/klsh/all_tests.jsonl'"
find research/eval/klsh | grep  '/test.*json$' | parallel 'cat {}' | jq -scM "sort_by(.f_score) | .[]" > "research/eval/klsh/all_tests.jsonl"

echo "success!"
