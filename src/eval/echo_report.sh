#!/bin/bash
set -e
all_models="$( cat research/eval/adahash/all_tests.jsonl research/eval/klsh/all_tests.jsonl research/eval/tlsh/all_tests.jsonl )"

models="$( echo "$all_models" | jq -r '.hparams.model' | sort | uniq )"
datasets="$( echo "$all_models" | jq -r '.hparams.database'| awk -F'-' '{print $1}' | sort | uniq )"

echo '# sorted by f-score'
echo
for dataset in $datasets
do
    echo
    echo
    echo "## $dataset"
    echo

    for model in $models
    do
        echo "### $model"
        echo
        echo

        best_hparams="$( \
            echo "$all_models" | \
            jq -csM "[.[] | select(.hparams.model == \"$model\" and (.hparams.database | startswith( \"$dataset\")))] | sort_by(.f_score) | reverse[]" | \
                head -n 40 \
        )"

        # echo header
        echo "$best_hparams" | jq -scMr '[.[:1][] |.hparams+{"prediction_type":.prediction_type,"reduction_ratio":.reduction_ratio, "recall":.recall, "precision":.precision, "f_score":.f_score} | to_entries[] | select(.key != "model" and .key != "database" and .key != "commit" ) | .key ] | join(" | ")'
        echo "$best_hparams" | jq -scMr '[.[:1][] |.hparams+{"prediction_type":.prediction_type,"reduction_ratio":.reduction_ratio, "recall":.recall, "precision":.precision, "f_score":.f_score} | to_entries[] | select(.key != "model" and .key != "database" and .key != "commit" ) | "---" ] | join(" | ")'
        # echo data
        echo "$best_hparams" | jq -cMr '[ .hparams+{"prediction_type":.prediction_type, "reduction_ratio":.reduction_ratio, "recall":.recall, "precision":.precision, "f_score":.f_score} | to_entries[] | select(.key != "model" and .key != "database" and .key != "commit" ) | .value]  | join(" | ")'
    done
done

echo '# sorted by reduction_ratio * recall'
echo
for dataset in $datasets
do
    echo
    echo
    echo "## $dataset"
    echo

    for model in $models
    do
        echo "### $model"
        echo
        echo

        best_hparams="$( \
            echo "$all_models" | \
            jq -csM "[.[] | select(.hparams.model == \"$model\" and (.hparams.database | startswith( \"$dataset\")))] | sort_by(.reduction_ratio * .recall) | reverse[]" | \
                head -n 40 \
        )"

        # echo header
        echo "$best_hparams" | jq -scMr '[.[:1][] |.hparams+{"prediction_type":.prediction_type,"reduction_ratio":.reduction_ratio, "recall":.recall, "precision":.precision, "rr*recall":(.reduction_ratio * .recall)} | to_entries[] | select(.key != "model" and .key != "database" and .key != "commit" ) | .key ] | join(" | ")'
        echo "$best_hparams" | jq -scMr '[.[:1][] |.hparams+{"prediction_type":.prediction_type,"reduction_ratio":.reduction_ratio, "recall":.recall, "precision":.precision, "rr*recall":(.reduction_ratio * .recall)} | to_entries[] | select(.key != "model" and .key != "database" and .key != "commit" ) | "---" ] | join(" | ")'
        # echo data
        echo "$best_hparams" | jq -cMr '[ .hparams+{"prediction_type":.prediction_type, "reduction_ratio":.reduction_ratio, "recall":.recall, "precision":.precision, "rr*recall":(.reduction_ratio * .recall)} | to_entries[] | select(.key != "model" and .key != "database" and .key != "commit" ) | .value]  | join(" | ")'
    done
done
