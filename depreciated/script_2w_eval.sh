#!/bin/bash
sourc/homes/c20beliv/Documents/Stage/knowledge_base_population/evale //users/local/c20beliv/myvenv/bin/activate
BASE=$(pwd)

cd src
echo "******* EVAL *******"

#SAVE_NAME="zero_shot_trained_large_v"
MODEL="microsoft/deberta-v3-base"

#MODEL="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
for SPLIT in 20; do

    SAVE_NAME="untrained_2w_derbertabase_biased_split"$SPLIT

    for i in  6 9; do
        python3 eval_2w_final.py \
            --input_file $BASE"/data/WN18RR/test_eval.json" \
            --input_file_indirect $BASE"/data/WN18RR/test_eval_indirect.json" \
            --output_file "eval" \
            --model "//users/local/c20beliv/model/$SAVE_NAME"_v"$i/MNLI/checkpoint-6500/" \
            --name "combined_"$SAVE_NAME"_v$i" \
            --source_model $MODEL \
            --normalise true
    done
done
#====================================================================++ 