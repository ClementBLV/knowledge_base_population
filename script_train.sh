#!/bin/bash
BASE=$(pwd)
GLUE_DIR=$BASE"/src/"
TASK_NAME="MNLI"

#libs/transformers/examples/pytorch/text-classification/run_glue.py 
python3 "src/run_glue.py" \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir "data/WN18RR/" \
  --max_seq_length "128" \
  --per_gpu_train_batch_size "32" \
  --learning_rate "2e-5" \
  --num_train_epochs "3.0" \
  --output_dir /tmp/$TASK_NAME/