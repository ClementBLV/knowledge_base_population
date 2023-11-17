#!/bin/bash
BASE=$(pwd)

cd src
echo "******* EVAL *******"

# "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"
MODEL=#"MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
SPLIT=1
SAVE_NAME="train_"$SPLIT"_train_dbl"

python3 eval.py \
    --input_file $BASE"/data/WN18RR/valid_eval.json" \
    --output_file "eval" \
    --model "/users/local/c20beliv/model/$SAVE_NAME/MNLI" \
    --name "train_"$SPLIT"_deberta_MoritzLaurer_base" \
    --source_model "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" \

