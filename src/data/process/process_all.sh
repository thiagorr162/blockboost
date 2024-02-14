#!/bin/bash

dataset=(
    abt_buy
    amazon_google
    dblp_acm
    dblp_scholar
    walmart_amazon
    restaurant
    rldata
)

shingling_sizes=(
    $( seq 1 6 )
)

parallel --halt-on-error 2 \
    'python src/data/process/process_{1}.py -sp .15 .70 ' \
    ::: "${dataset[@]}"
