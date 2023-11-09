#!/bin/bash
cd src

echo "******* EVAL *******"


python3 eval.py \
    --input_file "./data/WN18RR/valid_eval.json" \
    --output_file ".eval" \
    --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"
