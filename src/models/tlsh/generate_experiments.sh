#!/bin/bash

source src/utils/trap_errors.sh

ulimit -n `ulimit -Hn`

ulimit -v $(( 10 * 2 ** 30 ))

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
    musicbrainz_20
)


for d in "${list_of_datasets[@]}" ; do
    echo $d
done

seeds=( $( seq 0 3 ) )

n_buckets=(20 40 60 80 100 250 500 1000 1500)
shingle_size=( $( seq 1 6 ) )

start_time=$SECONDS

mkdir -p out/tlsh/predict
echo "Running predictions (outputs in out/tlsh/predict)..."
parallel -j100 --progress \
    'python src/models/tlsh/predict.py -d {1} --number_of_buckets {2} --shingle_size {3} --seed {4} &>> out/tlsh/predict/{#}.out' \
    ::: "${databases[@]}" \
    ::: "${n_buckets[@]}" \
    ::: "${shingle_size[@]}" \
    ::: "${seeds[@]}"
echo
elapsed=$(( SECONDS - start_time ))
echo
eval "echo Total prediction time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
echo
echo  "Evaluating runs"
./src/eval/eval.sh tlsh

echo
echo "Concatenating jsons of the test subset into 'research/eval/tlsh/all_tests.jsonl'"
find research/eval/tlsh | grep  '/test.*json$' | parallel 'cat {}' | jq -scM "sort_by(.f_score) | .[]" > "research/eval/tlsh/all_tests.jsonl"

echo "Success!"
