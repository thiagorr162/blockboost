#!/bin/bash

source src/utils/trap_errors.sh

ulimit -n `ulimit -Hn`

selected_model="adahash"

echo selected_model="'$selected_model'"

mkdir -p "out/eval/emb-restaurant-ctt-test"
mkdir -p "out/eval/knn-restaurant-ctt-test"

paths="$( find "research/models/adahash/restaurant-ctt-test" | grep  prediction.emb )"

paths_emb="$( find "research/models/adahash/restaurant-ctt-test" | grep  prediction.emb )"

start_time=$SECONDS
echo "number_of_predictions_emb=$( echo "$paths_emb" | wc -l )"
echo "Evaluating (stdout & stderr in out/eval/emb/)"
sleep 1
echo "$paths_emb" | parallel --halt-on-error 2 --progress \
    'python src/eval/embedding.py -o -p {} 2> out/eval/emb-restaurant-ctt-test/{#}.job' > "research/eval/adahash/restaurant-ctt-test_emb.jsonl" &

echo "number_of_predictions_emb=$( echo "$paths_emb" | wc -l )"
echo "Evaluating (stdout & stderr in out/eval/knn-restauran-ctt-te)"
sleep 1
echo "$paths_emb" | parallel --halt-on-error 2 --progress \
    'python src/eval/knn.py -o -p {} 2> out/eval/knn-restaurant-ctt-test/{#}.job' > "research/eval/adahash/restaurant-ctt-test_knn.jsonl"

elapsed=$(( SECONDS - start_time ))
echo
wait
eval "echo 'Total eval time (knn+emb):' $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
paths="$( echo "$paths" | grep prediction.bin )"
