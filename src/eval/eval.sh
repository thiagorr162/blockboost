#!/bin/bash

source src/utils/trap_errors.sh

ulimit -n `ulimit -Hn`

if (( $# != 1 ))
then
    echo './src/eval/eval.sh <model_name>'
    exit 1
fi

selected_model="$1"
echo selected_model="'$selected_model'"

mkdir -p "out/eval/emb"
mkdir -p "out/eval/knn"
mkdir -p "research/eval/$selected_model"

paths="$( find "research/models/$selected_model" | grep -e prediction.json -e prediction.bin -e prediction.emb )"

if [ "$selected_model" = "adahash" ] || [ "$selected_model" = "blockboost" ] || [ "$selected_model" = "blockboost-fp32" ] || [ $selected_model = "blockboost-fp32-save_train" ] ; then
    paths_emb="$( echo "$paths" | grep prediction.emb  )"

    if [ "$selected_model" = "blockboost-fp32" ] ; then
        # remove pn=8
        paths_emb="$( echo "$paths_emb" | grep -v pn_8  )"
    fi

    start_time=$SECONDS
    echo "number_of_predictions_emb=$( echo "$paths_emb" | wc -l )"
    echo "Evaluating (stdout & stderr in out/eval/emb/)"
    sleep 1
    echo "$paths_emb" | parallel --halt-on-error 2 --progress \
        'python src/eval/embedding.py -o -p {} 2> out/eval/emb/{#}.job' > "research/eval/$selected_model/all_emb.jsonl" &

    echo "number_of_predictions_emb=$( echo "$paths_emb" | wc -l )"
    echo "Evaluating (stdout & stderr in out/eval/knn/)"
    sleep 1
    echo "$paths_emb" | parallel --halt-on-error 2 --progress \
        'python src/eval/knn.py -o -p {} 2> out/eval/knn/{#}.job' > "research/eval/$selected_model/all_knn.jsonl"

    elapsed=$(( SECONDS - start_time ))
    echo
    wait
    eval "echo 'Total eval time:' $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"

    exit 0
fi


echo "number_of_predictions=$( echo "$paths" | wc -l )"
echo
echo "Evaluating (stdout & stderr in out/eval/)"
echo "Total jobs: $( echo "$paths" | wc -l )"
start_time=$SECONDS


if [ "$selected_model" = "adahash" ] ; then
    paths="$( echo "$paths" | grep prediction.bin | grep -v musicbrainz )"
    echo "$paths" | parallel --halt-on-error 2 --progress \
        'python src/eval/hamming.py -s -st -p {} &>> out/eval/{#}.job'
elif [ "$selected_model" = "canopy" ] ; then
    ulimit -v $(( 1024 * 1024 * 4 ))
    echo "$paths" | parallel -j128 --progress \
        'python src/eval/eval.py -p {} -s &>> out/eval/{#}.job'
else
    echo "$paths" | parallel --halt-on-error 2 --progress \
        'python src/eval/eval.py -p {} -s &>> out/eval/{#}.job'
fi


elapsed=$(( SECONDS - start_time ))
echo
eval "echo Total eval time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"

