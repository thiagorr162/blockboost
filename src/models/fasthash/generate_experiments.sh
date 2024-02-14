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
)

sizes=( $( seq 2 5 ) )


mkdir -p out/fasthash/predict

echo "Running predictions (outputs in out/fasthash/predict)..."

start_time=$SECONDS
parallel --halt-on-error 2  --progress \
    'python src/models/fasthash/predict.py -db {1} --size {2} &> out/fasthash/predict/{#}.out' \
    ::: "${databases[@]}" \
    ::: "${sizes[@]}"

parallel --halt-on-error 2  --progress \
    'python src/models/fasthash/predict.py -db {1} --size {2} --sort_hashes &> out/fasthash/predict/{#}.out' \
    ::: "${databases[@]}" \
    ::: "${sizes[@]}"

elapsed=$(( SECONDS - start_time ))
echo
eval "echo Total prediction time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
echo
echo  "Evaluating runs"
./src/eval/eval.sh fasthash

echo
echo "Concatenating jsons of the test subset into 'research/eval/fasthash/all_tests.jsonl'"
find research/eval/fasthash | grep  '/test.*json$' | parallel 'cat {}' | jq -scM "sort_by(.f_score) | .[]" > "research/eval/fasthash/all_tests.jsonl"

echo "success!"
