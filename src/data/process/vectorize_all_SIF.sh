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

parallel --max-procs 1 --halt-on-error 2 \
    'python src/data/process/vectorize_SIF.py -db {1}' \
    ::: "${dataset[@]}" \
