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

minhash_options=(
    vectorize_minhash-amber_ox-1
    vectorize_minhash-orange_tiger-3
    vectorize_minhash-tan_wildebeest-6
    vectorize_minhash-gray_bear-5
    vectorize_minhash-sapphire_jay-4
    vectorize_minhash-yellow_barracuda-2
)

parallel --max-procs 1 --halt-on-error 2 \
    'python src/data/process/vectorize_combine.py -v1 {1}-fasttext -v2 {1}-{2}' \
    ::: "${dataset[@]}" \
    ::: "${minhash_options[@]}" \
