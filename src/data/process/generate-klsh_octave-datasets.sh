#!/bin/bash

source src/utils/trap_errors.sh


datasets=(
    "restaurant-olive_limpet"
    "restaurant-gray_dingo"
    "restaurant-plum_peacock"
    "restaurant-beige_mockingbird"
    "restaurant-olive_limpet"
    "restaurant-gray_dingo"
    "restaurant-plum_peacock"
    "restaurant-beige_mockingbird"
    "RLDATA10000-coffee_mole"
    "RLDATA10000-green_wombat"
    "RLDATA10000-scarlet_mule"
    "RLDATA10000-turquoise_chinchilla"
    "RLDATA500-coffee_mole"
    "RLDATA500-green_wombat"
    "RLDATA500-scarlet_mule"
    "RLDATA500-turquoise_chinchilla"
    "cora"
    "cora_full"
#    "iris"
    "cifar10"
)
OUTPUT_DIR="research/data/processed-klsh_octave"

for d in "${datasets[@]}"
do
    for subset in "train" "test" "val"
    do
        if ls "research/data/processed/$d/$subset-vectorized.csv"
        then
            mkdir -p  "$OUTPUT_DIR/$d"

            pv "research/data/processed/$d/$subset-vectorized.csv" |\
                awk -F, 'NR>1 {for(i=1; i<NF-1;i++) {printf("%.8f ", $(i))}; print ""}' >\
                "$OUTPUT_DIR/$d/$subset.mtx"

            pv "research/data/processed/$d/$subset-vectorized.csv" |\
                awk -F, 'NR>1 {v[++i] = $(NF-1) ; if (! ($(NF-1) in vj)) {vj[$(NF-1)] = j++}} END {for (i in v) {print(vj[v[i]])}}' >\
                "$OUTPUT_DIR/$d/$subset.ids"

            echo
        else
            echo "skipping $d, $subset..."
        fi
    done
done
