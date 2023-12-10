#!/bin/bash
BASE=$(pwd)

cd src
echo "******* EVAL *******"

#"/users/local/c20beliv/model/$SAVE_NAME/MNLI" \
#SAVE_NAME="zero_shot_trained_large_v"

INDIRECT=false

MODEL="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
#MODEL="microsoft/deberta-v3-base"
for SPLIT in 7; do

    SAVE_NAME="trained_derbertabase_biased_split"$SPLIT"_v"

    for i in 6; do
        python3 eval.py \
            --input_file $BASE"/data/WN18RR/test_eval.json" \
            --output_file "eval" \
            --model "/users/local/c20beliv/model/$SAVE_NAME$i/MNLI/" \
            --name $SAVE_NAME"$i" \
            --source_model $MODEL 
        if $INDIRECT ; then 
            python3 eval.py \
                --input_file $BASE"/data/WN18RR/test_eval_indirect.json" \
                --output_file "eval" \
                --model "/users/local/c20beliv/model/$SAVE_NAME$i/MNLI/checkpoint-1000/" \
                --name $SAVE_NAME"_indirect_$i" \
                --source_model $MODEL 
        fi
    done
done