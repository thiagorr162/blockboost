#!/bin/bash

source src/utils/trap_errors.sh

ulimit -n `ulimit -Hn`

databases=(
    "abt_buy-ctt_3"
    "amazon_google-ctt_3"
    "dblp_acm-ctt_3"
    "dblp_scholar-ctt_3"
    #"musicbrainz_20-ctt_3"
    "restaurant-ctt_3"
    "RLDATA10000-ctt_3"
    "RLDATA500-ctt_3"
    "walmart_amazon-ctt_3"

    "RLDATA10000-fasttext"
    "RLDATA10000-vectorize_minhash-amber_ox-1"
    "RLDATA10000-vectorize_minhash-gray_bear-5"
    "RLDATA10000-vectorize_minhash-orange_tiger-3"
    "RLDATA10000-vectorize_minhash-sapphire_jay-4"
    "RLDATA10000-vectorize_minhash-tan_wildebeest-6"
    "RLDATA10000-vectorize_minhash-yellow_barracuda-2"
    "RLDATA500-fasttext"
    "RLDATA500-vectorize_minhash-amber_ox-1"
    "RLDATA500-vectorize_minhash-gray_bear-5"
    "RLDATA500-vectorize_minhash-orange_tiger-3"
    "RLDATA500-vectorize_minhash-sapphire_jay-4"
    "RLDATA500-vectorize_minhash-tan_wildebeest-6"
    "RLDATA500-vectorize_minhash-yellow_barracuda-2"
    "abt_buy-fasttext"
    "abt_buy-vectorize_minhash-amber_ox-1"
    "abt_buy-vectorize_minhash-gray_bear-5"
    "abt_buy-vectorize_minhash-orange_tiger-3"
    "abt_buy-vectorize_minhash-sapphire_jay-4"
    "abt_buy-vectorize_minhash-tan_wildebeest-6"
    "abt_buy-vectorize_minhash-yellow_barracuda-2"
    "amazon_google-fasttext"
    "amazon_google-vectorize_minhash-amber_ox-1"
    "amazon_google-vectorize_minhash-gray_bear-5"
    "amazon_google-vectorize_minhash-orange_tiger-3"
    "amazon_google-vectorize_minhash-sapphire_jay-4"
    "amazon_google-vectorize_minhash-tan_wildebeest-6"
    "amazon_google-vectorize_minhash-yellow_barracuda-2"
    "dblp_acm-fasttext"
    "dblp_acm-vectorize_minhash-amber_ox-1"
    "dblp_acm-vectorize_minhash-gray_bear-5"
    "dblp_acm-vectorize_minhash-orange_tiger-3"
    "dblp_acm-vectorize_minhash-sapphire_jay-4"
    "dblp_acm-vectorize_minhash-tan_wildebeest-6"
    "dblp_acm-vectorize_minhash-yellow_barracuda-2"
    "dblp_scholar-fasttext"
    "dblp_scholar-vectorize_minhash-amber_ox-1"
    "dblp_scholar-vectorize_minhash-gray_bear-5"
    "dblp_scholar-vectorize_minhash-orange_tiger-3"
    "dblp_scholar-vectorize_minhash-sapphire_jay-4"
    "dblp_scholar-vectorize_minhash-tan_wildebeest-6"
    "dblp_scholar-vectorize_minhash-yellow_barracuda-2"
    "restaurant-fasttext"
    "restaurant-vectorize_minhash-amber_ox-1"
    "restaurant-vectorize_minhash-gray_bear-5"
    "restaurant-vectorize_minhash-orange_tiger-3"
    "restaurant-vectorize_minhash-sapphire_jay-4"
    "restaurant-vectorize_minhash-tan_wildebeest-6"
    "restaurant-vectorize_minhash-yellow_barracuda-2"
    "walmart_amazon-fasttext"
    "walmart_amazon-vectorize_minhash-amber_ox-1"
    "walmart_amazon-vectorize_minhash-gray_bear-5"
    "walmart_amazon-vectorize_minhash-orange_tiger-3"
    "walmart_amazon-vectorize_minhash-sapphire_jay-4"
    "walmart_amazon-vectorize_minhash-tan_wildebeest-6"
    "walmart_amazon-vectorize_minhash-yellow_barracuda-2"
    "musicbrainz_20-fasttext"
    "musicbrainz_20-vectorize_minhash-amber_ox-1"
    "musicbrainz_20-vectorize_minhash-gray_bear-5"
    "musicbrainz_20-vectorize_minhash-orange_tiger-3"
    "musicbrainz_20-vectorize_minhash-sapphire_jay-4"
    "musicbrainz_20-vectorize_minhash-tan_wildebeest-6"
    "musicbrainz_20-vectorize_minhash-yellow_barracuda-2"
    #"musicbrainz_200-fasttext"
    #"musicbrainz_200-vectorize_minhash-amber_ox-1"
    #"musicbrainz_200-vectorize_minhash-gray_bear-5"
    #"musicbrainz_200-vectorize_minhash-orange_tiger-3"
    #"musicbrainz_200-vectorize_minhash-sapphire_jay-4"
    #"musicbrainz_200-vectorize_minhash-tan_wildebeest-6"
    #"musicbrainz_200-vectorize_minhash-yellow_barracuda-2"

    "RLDATA10000-fasttext-vectorize_minhash-amber_ox-1"
    "RLDATA10000-fasttext-vectorize_minhash-gray_bear-5"
    "RLDATA10000-fasttext-vectorize_minhash-orange_tiger-3"
    "RLDATA10000-fasttext-vectorize_minhash-sapphire_jay-4"
    "RLDATA10000-fasttext-vectorize_minhash-tan_wildebeest-6"
    "RLDATA10000-fasttext-vectorize_minhash-yellow_barracuda-2"
    "RLDATA500-fasttext-vectorize_minhash-amber_ox-1"
    "RLDATA500-fasttext-vectorize_minhash-gray_bear-5"
    "RLDATA500-fasttext-vectorize_minhash-orange_tiger-3"
    "RLDATA500-fasttext-vectorize_minhash-sapphire_jay-4"
    "RLDATA500-fasttext-vectorize_minhash-tan_wildebeest-6"
    "RLDATA500-fasttext-vectorize_minhash-yellow_barracuda-2"
    "abt_buy-fasttext-vectorize_minhash-amber_ox-1"
    "abt_buy-fasttext-vectorize_minhash-gray_bear-5"
    "abt_buy-fasttext-vectorize_minhash-orange_tiger-3"
    "abt_buy-fasttext-vectorize_minhash-sapphire_jay-4"
    "abt_buy-fasttext-vectorize_minhash-tan_wildebeest-6"
    "abt_buy-fasttext-vectorize_minhash-yellow_barracuda-2"
    "amazon_google-fasttext-vectorize_minhash-amber_ox-1"
    "amazon_google-fasttext-vectorize_minhash-gray_bear-5"
    "amazon_google-fasttext-vectorize_minhash-orange_tiger-3"
    "amazon_google-fasttext-vectorize_minhash-sapphire_jay-4"
    "amazon_google-fasttext-vectorize_minhash-tan_wildebeest-6"
    "amazon_google-fasttext-vectorize_minhash-yellow_barracuda-2"
    "dblp_acm-fasttext-vectorize_minhash-amber_ox-1"
    "dblp_acm-fasttext-vectorize_minhash-gray_bear-5"
    "dblp_acm-fasttext-vectorize_minhash-orange_tiger-3"
    "dblp_acm-fasttext-vectorize_minhash-sapphire_jay-4"
    "dblp_acm-fasttext-vectorize_minhash-tan_wildebeest-6"
    "dblp_acm-fasttext-vectorize_minhash-yellow_barracuda-2"
    "dblp_scholar-fasttext-vectorize_minhash-amber_ox-1"
    "dblp_scholar-fasttext-vectorize_minhash-gray_bear-5"
    "dblp_scholar-fasttext-vectorize_minhash-orange_tiger-3"
    "dblp_scholar-fasttext-vectorize_minhash-sapphire_jay-4"
    "dblp_scholar-fasttext-vectorize_minhash-tan_wildebeest-6"
    "dblp_scholar-fasttext-vectorize_minhash-yellow_barracuda-2"
    "restaurant-fasttext-vectorize_minhash-amber_ox-1"
    "restaurant-fasttext-vectorize_minhash-gray_bear-5"
    "restaurant-fasttext-vectorize_minhash-orange_tiger-3"
    "restaurant-fasttext-vectorize_minhash-sapphire_jay-4"
    "restaurant-fasttext-vectorize_minhash-tan_wildebeest-6"
    "restaurant-fasttext-vectorize_minhash-yellow_barracuda-2"
    "walmart_amazon-fasttext-vectorize_minhash-amber_ox-1"
    "walmart_amazon-fasttext-vectorize_minhash-gray_bear-5"
    "walmart_amazon-fasttext-vectorize_minhash-orange_tiger-3"
    "walmart_amazon-fasttext-vectorize_minhash-sapphire_jay-4"
    "walmart_amazon-fasttext-vectorize_minhash-tan_wildebeest-6"
    "walmart_amazon-fasttext-vectorize_minhash-yellow_barracuda-2"
    "musicbrainz_20-fasttext-vectorize_minhash-amber_ox-1"
    "musicbrainz_20-fasttext-vectorize_minhash-gray_bear-5"
    "musicbrainz_20-fasttext-vectorize_minhash-orange_tiger-3"
    "musicbrainz_20-fasttext-vectorize_minhash-sapphire_jay-4"
    "musicbrainz_20-fasttext-vectorize_minhash-tan_wildebeest-6"
    "musicbrainz_20-fasttext-vectorize_minhash-yellow_barracuda-2"
    #"musicbrainz_200-vectorize_minhash-amber_ox-1"
    #"musicbrainz_200-vectorize_minhash-gray_bear-5"
    #"musicbrainz_200-vectorize_minhash-orange_tiger-3"
    #"musicbrainz_200-vectorize_minhash-sapphire_jay-4"
    #"musicbrainz_200-vectorize_minhash-tan_wildebeest-6"
    #"musicbrainz_200-vectorize_minhash-yellow_barracuda-2"
)

seeds=( $( seq 0 3 ) )

for d in "${list_of_datasets[@]}" ; do
    echo $d
done

#proportion_nonmatches=(1 2 5 10 50 100 200)
proportion_nonmatches=( 8 4 2 1 )

mkdir -p out/blockboost/train
echo "Training (outputs in out/blockboost/train)..."

start_time=$SECONDS
parallel --halt-on-error 2 -j10 --progress \
    './bin/models/blockboost/train {2} {3} 1 {1} 5000 &> out/blockboost/train/{#}.job' \
    ::: "${proportion_nonmatches[@]}" \
    ::: "${databases[@]}" \
    ::: "${seeds[@]}" \


elapsed=$(( SECONDS - start_time ))
echo
eval "echo Total training time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"
echo

#echo
#echo  "Evaluating runs"
#src/eval/eval.sh blockboost

echo "Success!"

