#!/bin/bash

source src/utils/trap_errors.sh

python src/models/manual/predict.py -db musicbrainz_20 --features "number" "length"
python src/eval/eval.py -p research/models/manual/musicbrainz_20-number-length/test-full_prediction.json
echo "success!"
