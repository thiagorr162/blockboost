#!/bin/bash

echo_entity_ids() {
    dataset=$1
    cat research/data/processed/"$dataset"/*.csv | awk -F, 'NR>1 {print $(NF-1)}'| sed "s/'//g"
}

OUTPUT_DIR="research/data/eval"

make
echo
mkdir -p "$OUTPUT_DIR"

for dataset in restaurant RLDATA500 RLDATA10000 cora cora_full ; do
    echo "processing $dataset"
    echo_entity_ids "$dataset" | bin/number_of_matches_of_subsets 1000 > "$OUTPUT_DIR/$dataset-number_of_matches_of_subsets.txt"
    echo_entity_ids "$dataset" | bin/number_of_matches_of_subsets-over-comb 1000 > "$OUTPUT_DIR/$dataset-number_of_matches_of_subsets-over-comb.txt"
    echo "generating plots for $dataset"
    cat "$OUTPUT_DIR/$dataset-number_of_matches_of_subsets.txt" | plt --pdf --label "$dataset: number of matches for subset with size ( len($dataset) * x / 100 )" > "$OUTPUT_DIR/$dataset-number_of_matches_of_subsets.pdf"
    cat "$OUTPUT_DIR/$dataset-number_of_matches_of_subsets-over-comb.txt" | plt --pdf --label "$dataset: number of matches divided by Comb(k, 2)" > "$OUTPUT_DIR/$dataset-number_of_matches_of_subsets-over-comb.pdf"
    echo ""
done
