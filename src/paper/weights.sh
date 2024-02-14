#!/usr/bin/env bash

mkdir -p research/paper-blockboost/weights/

./src/data/process/split_columns-musicbrainz.sh
python src/data/process/vectorize_combine_split.py
./bin/models/blockboost-fp32/train-omp musicbrainz_20-combined-a0dc5acac3ee3ed0 0 4 16 5000
cp research/models/blockboost-fp32/musicbrainz_20-combined-a0dc5acac3ee3ed0/s_0-pm_4-pn_16-n_5000/embedding/hparams.json \
    research/paper-blockboost/weights

python src/eval/weights_of_split_dataset.py > research/paper-blockboost/weights/weights_per_field.json

mkdir research/paper-blockboost/weights/eval
python src/eval/embedding.py -o -p research/models/blockboost-fp32/musicbrainz_20-combined-a0dc5acac3ee3ed0/s_0-pm_4-pn_16-n_5000/embedding/val_prediction.emb > research/paper-blockboost/weights/eval/val.jsonl
python src/eval/embedding.py -o -p research/models/blockboost-fp32/musicbrainz_20-combined-a0dc5acac3ee3ed0/s_0-pm_4-pn_16-n_5000/embedding/test_prediction.emb > research/paper-blockboost/weights/eval/test.jsonl

