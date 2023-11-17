#!/bin/bash
BASE=$(pwd)
GLUE_DIR=$BASE"/src/"
TASK_NAME="MNLI"

MODEL="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

for SPLIT in 1; do
  start_time=$(date +%s.%N)

  echo "******* TRAIN *******"
  ##--task_name $TASK_NAME \
  #libs/transformers/examples/pytorch/text-classification/run_glue.py
  python3 "src/run_glue.py" \
    --model_name_or_path "$MODEL" \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file "$BASE/data/WN18RR/train_$SPLIT.mnli.json" \
    --test_file "$BASE/data/WN18RR/test.mnli.json" \
    --validation_file "$BASE/data/WN18RR/valid.mnli.json" \
    --max_seq_length "128" \
    --per_gpu_train_batch_size "1" \
    --learning_rate "2e-5" \
    --num_train_epochs "3" \
    --output_dir "/users/local/c20beliv/tmp/$TASK_NAME/" \
    --save_total_limit "1" \
    --ignore_mismatched_sizes "True" \


  echo "******* TIME *******"
  end_time=$(date +%s.%N)
  # Calculate execution time in seconds
  execution_time=$(echo "$end_time - $start_time" | bc)

  # Calculate hours, minutes, and seconds
  hours=$(echo "$execution_time / 3600" | bc)
  minutes=$(echo "($execution_time % 3600) / 60" | bc)
  seconds=$(echo "$execution_time % 60" | bc)
  echo "Execution time: ${hours}h ${minutes}m ${seconds}s"


  # save 
  SAVE_NAME="train_"$SPLIT"_train_dbl"

  NEW_FILE="/users/local/c20beliv/model/$SAVE_NAME"
  mkdir $NEW_FILE
  mv "/users/local/c20beliv/tmp/$TASK_NAME" $NEW_FILE
done
