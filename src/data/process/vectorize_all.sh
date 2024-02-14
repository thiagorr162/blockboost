#!/bin/bash

dataset=(
    abt_buy
    amazon_google
    dblp_acm
    dblp_scholar
    walmart_amazon
    restaurant
    RLDATA500
    RLDATA10000
    musicbrainz_20
    musicbrainz_200
)

shingling_sizes=(
    $( seq 1 6 )
)

parallel -k --halt-on-error 2 \
    'python src/data/process/vectorize_minhash.py -d {1} -k {2} --shared 2>&1 | grep OUTPUT_DIR' \
    ::: "${dataset[@]}" \
    ::: "${shingling_sizes[@]}"
