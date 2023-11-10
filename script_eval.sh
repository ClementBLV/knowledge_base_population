#!/bin/bash
cd src
BASE=$(pwd)
echo "******* EVAL *******"


python3 eval.py \
    --input_file $BASE"/data/WN18RR/valid_eval.json" \
    --output_file $BASE"/eval" \
    --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"
