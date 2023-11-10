#!/bin/bash
BASE=$(pwd)
GLUE_DIR=$BASE"/src/"
TASK_NAME="MNLI"

start_time=$(date +%s.%N)

echo "******* TRAIN *******"

#libs/transformers/examples/pytorch/text-classification/run_glue.py
python3 "src/run_glue.py" \
  --model_name_or_path "microsoft/deberta-v3-small" \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --train_file "$BASE/data/WN18RR/train_1.mnli.json" \
  --test_file "$BASE/data/WN18RR/test.mnli.json" \
  --validation_file "$BASE/data/WN18RR/valid.mnli.json" \
  --max_seq_length "128" \
  --per_gpu_train_batch_size "25" \
  --learning_rate "2e-5" \
  --num_train_epochs "1" \
  --output_dir "$BASE/tmp/$TASK_NAME/" \
  --save_total_limit "1" \

end_time=$(date +%s.%N)
execution_time=$(echo "$end_time - $start_time" | bc)
echo "Execution time: $execution_time seconds"
