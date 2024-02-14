#!/usr/bin/env bash

source src/utils/trap_errors.sh

INPUT_DIR="research/data/processed/musicbrainz_20"

IDX="$( head -n1 "$INPUT_DIR/train-textual.csv" | awk -F, '{for (i = 1; i<=NF-2; i++) {print i} }' )"
rm -rf "$INPUT_DIR-split-"*
OUTPUT_DIRS=()

for ID in $IDX ; do
    FIELD="$( head -n1 "$INPUT_DIR/train-textual.csv" | awk -F, "{printf \"%s\", \$($ID)}" )"

    OUTPUT_DIR="$INPUT_DIR-split-$FIELD"
    OUTPUT_DIRS+=($OUTPUT_DIR)
    echo "$ID,$FIELD"
    mkdir -p "$OUTPUT_DIR"
    cp "$INPUT_DIR"/*.json "$OUTPUT_DIR"

    for CSV_PATH in "$INPUT_DIR"/*.csv ; do
        CSV_FILENAME="$( basename "$CSV_PATH" )"


        # filter entries with more than 8 commas
        # Example of removed line:
        #   "first, field",field 2,...
        cat "$CSV_PATH"|\
            grep -v '.*,.*,.*,.*,.*,.*,.*,.*,.*,' |\
            awk  -F, "{print \$($ID) \" 0000000000,\" \$(NF-1) \",\" \$(NF)}" > "$OUTPUT_DIR/$CSV_FILENAME"
    done
done

echo "${OUTPUT_DIRS[@]}"


shingling_sizes=( 4 )

parallel -k --halt-on-error 2 \
    'python src/data/process/vectorize_minhash.py -d "$( basename {1} )" -k {2} --shared 2>&1' \
    ::: "${OUTPUT_DIRS[@]}" \
    ::: "${shingling_sizes[@]}"


