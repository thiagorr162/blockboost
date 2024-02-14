#!/bin/bash

source src/utils/trap_errors.sh


ulimit -v $(( 1024 * 1024 * 8 ))


databases_textual=(
    "RLDATA10000"
    "RLDATA500"
    "abt_buy"
    "amazon_google"
    "dblp_acm"
    "dblp_scholar"
    "restaurant"
    "walmart_amazon"
    "musicbrainz_20"
)

seeds=( $( seq 0 1 ) )

for d in "${list_of_datasets[@]}" ; do
    echo $d
done

t1_list=( $( seq .35 .2 .95 ) )
t2_list=( $( seq .35 .2 .95 ) )
vect_dist_list=("euclidean" "jaccard" )
text_dist_list=("jaccard-mean" "jaccard-concat" )

mkdir -p out/canopy/predict
echo "Running predictions (stdout & stderr in out/canopy/predict)..."
start_time=$SECONDS

#echo "Total number of vectorized data predictions: $(( ${#databases_vectorized[@]} * ${#t1_list[@]} * ${#t2_list[@]} * ${#vect_dist_list[@]} *${#seeds[@]} ))"
#
#parallel -j 1 --halt-on-error 2 --progress \
#    'python src/models/canopy/predict.py -db {1} -dt vectorized -dist {2} -t1 {3} -t2 {4} -s {5} &>> out/canopy/predict/{#}.job' \
#    ::: "${databases_vectorized[@]}" \
#    ::: "${vect_dist_list[@]}" \
#    ::: "${t1_list[@]}" \
#    ::: "${t2_list[@]}" \
#    ::: "${seeds[@]}"
#
echo "Total number of textual data predictions: $(( ${#databases_textual[@]} * ${#t1_list[@]} * ${#t2_list[@]} * ${#text_dist_list[@]} *${#seeds[@]} ))"

parallel -j 64 --progress \
    'python src/models/canopy/predict.py -db {1} -dt textual -dist {2} -t1 {3} -t2 {4} -s {5} &>> out/canopy/predict/{#}.job' \
    ::: "${databases_textual[@]}" \
    ::: "${text_dist_list[@]}" \
    ::: "${t1_list[@]}" \
    ::: "${t2_list[@]}" \
    ::: "${seeds[@]}"

elapsed=$(( SECONDS - start_time ))
echo
eval "echo Total prediction time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"

echo
echo  "Evaluating runs"
src/eval/eval.sh canopy
echo
echo "Concatenating jsons of the test subset into 'research/eval/canopy/all_tests.jsonl'"
find research/eval/canopy | grep  '/test.*json$' | parallel 'cat {}' | jq -scM "sort_by(.f_score) | .[]" > "research/eval/canopy/all_tests.jsonl"

echo "Success!"


