#!/bin/bash
BASE=$(pwd)

cd src
echo "******* EVAL *******"

#"/users/local/c20beliv/model/$SAVE_NAME/MNLI" \
#SAVE_NAME="zero_shot_trained_large_v"


MODEL="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
#MODEL="microsoft/deberta-v3-base"
for SPLIT in 20; do

    SAVE_NAME="trained_derbertabase_biased_split"$SPLIT"_v"

    for i in 3; do
        python3 eval.py \
            --input_file $BASE"/data/WN18RR/test_eval.json" \
            --output_file "eval" \
            --model "/users/local/c20beliv/model/$SAVE_NAME$i/MNLI/" \
            --name $SAVE_NAME"$i" \
            --source_model $MODEL 
    done
done