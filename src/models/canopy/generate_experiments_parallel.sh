#!/bin/bash

source src/utils/trap_errors.sh

databases=(
    RLDATA500-vectorize_minhash-from-tlsh-jade_tarsier-6
    RLDATA500-vectorize_minhash-from-tlsh-ivory_tortoise-1
    RLDATA500-vectorize_minhash-from-tlsh-pink_catfish-2
    RLDATA500-vectorize_minhash-from-tlsh-magenta_fowl-3
    RLDATA500-vectorize_minhash-from-tlsh-harlequin_catfish-4
    RLDATA500-vectorize_minhash-from-tlsh-tan_hedgehog-5
    restaurant-vectorize_minhash-from-tlsh-magenta_fowl-3
    restaurant-vectorize_minhash-from-tlsh-ivory_tortoise-1
    restaurant-vectorize_minhash-from-tlsh-pink_catfish-2
    restaurant-vectorize_minhash-from-tlsh-harlequin_catfish-4
    restaurant-vectorize_minhash-from-tlsh-jade_tarsier-6
    restaurant-vectorize_minhash-from-tlsh-tan_hedgehog-5
    RLDATA10000-vectorize_minhash-from-tlsh-pink_catfish-2
    RLDATA10000-vectorize_minhash-from-tlsh-ivory_tortoise-1
    RLDATA10000-vectorize_minhash-from-tlsh-magenta_fowl-3
    RLDATA10000-vectorize_minhash-from-tlsh-harlequin_catfish-4
    RLDATA10000-vectorize_minhash-from-tlsh-tan_hedgehog-5
    RLDATA10000-vectorize_minhash-from-tlsh-jade_tarsier-6
)

for d in "${databases[@]}" ; do
    echo $d
done

t1_list=(0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95)
t2_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

start_time=$SECONDS

parallel --halt-on-error 2 --progress \
    'python src/models/canopy/train.py -db {1} -t1 {2} -t2 {3} -dist jaccard -dt vectorized &>> out/canopy/vectorized/{#}.job' \    
    ::: "${databases[@]}" \
    ::: "${t1_list[@]}" \
    ::: "${t2_list[@]}"

elapsed=$(( SECONDS - start_time ))
echo
eval "echo Total prediction time on vectorized datasets: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"

echo
echo "Evaluation runs"
src/eval/eval.sh canopy

echo "Sucess!"

#######################################

databases_textual=("restaurant" "RLDATA500" "RLDATA10000")

for d in "${databases_textual[@]}" ; do
    echo $d
done


distances=("jaccard-mean" "jaccard-concat")

start_time=$SECONDS

parallel --halt-on-error 2 --progress \
    'python src/models/canopy/train.py -db {1} -t1 {2} -t2 {3} -dist {4} -dt textual &>> out/canopy/textual/{#}.job' \        
    ::: "${databases_textual[@]}" \
    ::: "${t1_list[@]}" \
    ::: "${t2_list[@]}" \
    ::: "${distances[@]}"

elapsed=$(( SECONDS - start_time ))
echo
eval "echo Total prediction time on textual datasets: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"                                        


