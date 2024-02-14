#!/bin/bash

source src/utils/trap_errors.sh

databases=(
    "RLDATA10000-vectorize_minhash-amber_ox-1"
    "RLDATA10000-vectorize_minhash-gray_bear-5"
    "RLDATA10000-vectorize_minhash-orange_tiger-3"
    "RLDATA10000-vectorize_minhash-sapphire_jay-4"
    "RLDATA10000-vectorize_minhash-tan_wildebeest-6"
    "RLDATA10000-vectorize_minhash-yellow_barracuda-2"
    "RLDATA500-vectorize_minhash-amber_ox-1"
    "RLDATA500-vectorize_minhash-gray_bear-5"
    "RLDATA500-vectorize_minhash-orange_tiger-3"
    "RLDATA500-vectorize_minhash-sapphire_jay-4"
    "RLDATA500-vectorize_minhash-tan_wildebeest-6"
    "RLDATA500-vectorize_minhash-yellow_barracuda-2"
    "abt_buy-vectorize_minhash-amber_ox-1"
    "abt_buy-vectorize_minhash-gray_bear-5"
    "abt_buy-vectorize_minhash-orange_tiger-3"
    "abt_buy-vectorize_minhash-sapphire_jay-4"
    "abt_buy-vectorize_minhash-tan_wildebeest-6"
    "abt_buy-vectorize_minhash-yellow_barracuda-2"
    "amazon_google-vectorize_minhash-amber_ox-1"
    "amazon_google-vectorize_minhash-gray_bear-5"
    "amazon_google-vectorize_minhash-orange_tiger-3"
    "amazon_google-vectorize_minhash-sapphire_jay-4"
    "amazon_google-vectorize_minhash-tan_wildebeest-6"
    "amazon_google-vectorize_minhash-yellow_barracuda-2"
    "dblp_acm-vectorize_minhash-amber_ox-1"
    "dblp_acm-vectorize_minhash-gray_bear-5"
    "dblp_acm-vectorize_minhash-orange_tiger-3"
    "dblp_acm-vectorize_minhash-sapphire_jay-4"
    "dblp_acm-vectorize_minhash-tan_wildebeest-6"
    "dblp_acm-vectorize_minhash-yellow_barracuda-2"
    "dblp_scholar-vectorize_minhash-amber_ox-1"
    "dblp_scholar-vectorize_minhash-gray_bear-5"
    "dblp_scholar-vectorize_minhash-orange_tiger-3"
    "dblp_scholar-vectorize_minhash-sapphire_jay-4"
    "dblp_scholar-vectorize_minhash-tan_wildebeest-6"
    "dblp_scholar-vectorize_minhash-yellow_barracuda-2"
    "restaurant-vectorize_minhash-amber_ox-1"
    "restaurant-vectorize_minhash-gray_bear-5"
    "restaurant-vectorize_minhash-orange_tiger-3"
    "restaurant-vectorize_minhash-sapphire_jay-4"
    "restaurant-vectorize_minhash-tan_wildebeest-6"
    "restaurant-vectorize_minhash-yellow_barracuda-2"
    "walmart_amazon-vectorize_minhash-amber_ox-1"
    "walmart_amazon-vectorize_minhash-gray_bear-5"
    "walmart_amazon-vectorize_minhash-orange_tiger-3"
    "walmart_amazon-vectorize_minhash-sapphire_jay-4"
    "walmart_amazon-vectorize_minhash-tan_wildebeest-6"
    "walmart_amazon-vectorize_minhash-yellow_barracuda-2"
    #"cora"
    #"cora_full"
    #"cifar10"
)

proportions_of_matches=( $( seq .5 .05 1 ) )


mkdir -p out/threshold/predict

echo "Running predictions (outputs in out/threshold/predict)..."

start_time=$SECONDS
parallel -j 50% --halt-on-error 2  --progress \
    'python src/models/threshold/predict.py -db {1} -pm {2} &> out/threshold/predict/{#}.out' \
    ::: "${databases[@]}" \
    ::: "${proportions_of_matches[@]}"

elapsed=$(( SECONDS - start_time ))
echo
eval "echo Total prediction time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
echo
echo  "Evaluating runs"
./src/eval/eval.sh threshold

echo
echo "Concatenating jsons of the test subset into 'research/eval/threshold/all_tests.jsonl'"
find research/eval/threshold | grep  '/test.*json$' | parallel 'cat {}' | jq -scM "sort_by(.f_score) | .[]" > "research/eval/threshold/all_tests.jsonl"

echo "success!"
