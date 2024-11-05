#!/bin/bash
source //users/local/c20beliv/myvenv/bin/activate
BASE=$(pwd)

cd src
echo "******* EVAL *******"

#"/users/local/c20beliv/model/$SAVE_NAME/MNLI" \
#SAVE_NAME="zero_shot_trained_large_v"

INDIRECT=false

#MODEL="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MODEL="microsoft/deberta-v3-base"

for SPLIT in 7; do

    SAVE_NAME="untrained_2w_derbertabase_biased_split"$SPLIT
#checkpoint-1500/
    for i in 8; do
        echo "Split $SPLIT v $i"
        python3 eval.py \
            --input_file $BASE"/data/WN18RR/test_eval.json" \
            --output_file "eval" \
            --model "//users/local/c20beliv/model/$SAVE_NAME"_v"$i/MNLI/checkpoint-1500" \
            --name $SAVE_NAME"_v"$i \
            --source_model $MODEL
        if $INDIRECT ; then 
            python3 eval.py \
                --input_file $BASE"/data/WN18RR/test_eval_indirect.json" \
                --output_file "eval" \
                --model "/users/local/c20beliv/model/$SAVE_NAME"_v"$i/MNLI/checkpoint-1000/" \
                --name $SAVE_NAME"_indirect_v$i" \
                --source_model $MODEL 
        fi
    done
done
#====================================================================++ 
INDIRECT=false

MODEL="microsoft/deberta-v3-base"
for SPLIT in 7; do

    SAVE_NAME="untrained_derbertabase_biased_split"$SPLIT"_v"
#checkpoint-1500/
    for i in 1000 ; do
        python3 eval.py \
            --input_file $BASE"/data/WN18RR/test_eval.json" \
            --output_file "eval" \
            --model "/users/local/c20beliv/model/$SAVE_NAME$i/MNLI/checkpoint-1000/" \
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