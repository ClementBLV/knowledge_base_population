#!/bin/bash
source //users/local/c20beliv/myvenv/bin/activate
BASE=$(pwd)
GLUE_DIR=$BASE"/src/"
TASK_NAME="MNLI"

# wandb
MY_KEY="d454d24f48a094d782b4f38707b2134a3dee9c40"
MY_PROJECT_NAME="knowledge-base-v5"
ENTITY="clementblv"
export WANDB_ENTITY=$ENTITY
export WANDB_PROJECT=$MY_PROJECT_NAME
export WANDB_DIR="/users/local/c20beliv/"
export WANDB_CACHE_DIR="/users/local/c20beliv/"

# hugging face
export HF_HOME="/users/local/c20beliv/hf/"
export HF_DATASETS_CACHE="/users/local/c20beliv/hf/datasets"
export TRANSFORMERS_CACHE="/users/local/c20beliv/hf/models"


# Models possible
#MODEL="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MODEL="microsoft/deberta-v3-base"
#MODEL="//users/local/c20beliv/model/untrained_derbertabase_biased_split20_v1/MNLI/checkpoint-1500"

BIAS=true
BOTH=false
echo $BIAS
for i in 9 10 ; do
  for SPLIT in 7; do
    ## TO MODIFY
    if ! $BIAS; then 
      echo $BIAS
      if [[ "$MODEL" == 'microsoft/deberta-v3-base' ]]; then 
        echo "UNTRAINED + UNBIAS!!" 
        SAVE_NAME="untrained_2w_derbertabase_unbiased_split"$SPLIT"_v$i"
        export WANDB_RUN_NAME="Deberta-naive-2w-UnB-$SPLIT-v$i"

      fi
      if [[ "$MODEL" == 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli' ]]; then 
        echo "TRAINED + UNBIAS!!" 
        SAVE_NAME="trained_2w_derbertabase_unbiased_split"$SPLIT"_v$i"
        export WANDB_RUN_NAME="Deberta-mnli-2w-UnB-$SPLIT-v$i"

      fi
    fi
    if $BIAS; then 
      if [[ "$MODEL" == 'microsoft/deberta-v3-base' ]]; then 
        echo "UNTRAINED + BIAS!!" 
        SAVE_NAME="untrained_2w_derbertabase_biased_split"$SPLIT"_v$i"
        export WANDB_RUN_NAME="Deberta-naive-2w-B-$SPLIT-v$i"

      fi
      if [[ "$MODEL" == 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli' ]]; then 
        echo "TRAINED + BIAS!!" 
        SAVE_NAME="trained_2w_derbertabase_biased_split"$SPLIT"_v$i"
        export WANDB_RUN_NAME="Deberta-mnli-2w-B-$SPLIT-v$i"

      fi
    fi
    # PATH of the data
    #P_FILE=$BASE"/data/WN18RR"
    P_FILE="/users/local/c20beliv/"

    ## wandb directory

    # Set the run name using the WandB command line tool
    #export WANDB_RUN_NAME="Deberta-naive-2w-$SPLIT-v$i"

    # remove the old datasets
    echo "remove old files"
    rm -rf $P_FILE"/train_"$SPLIT".json"
    rm -rf $P_FILE"/train_"$SPLIT".mnli.json"

    # generate a new dataset which will contain new triplets
    sh "script.sh" --splits $SPLIT --both $BOTH --bias $BIAS

    start_time=$(date +%s.%N)
    touch "/users/local/c20beliv/train.log" #create log file
    echo "=========== TRAIN ============"
    WANDB_API_KEY=$MY_KEY WANDB_PROJECT=$MY_PROJECT_NAME WANDB_ENTITY=$ENTITY python3 "src/run_glue.py" \
      --split $SPLIT\
      --model_name_or_path "$MODEL" \
      --do_train \
      --do_eval \
      --do_predict \
      --train_file $P_FILE"/train_$SPLIT.mnli.json" \
      --test_file $BASE"/data/WN18RR/test.mnli.json" \
      --validation_file $BASE"/data/WN18RR/valid.mnli.json" \
      --max_seq_length "128" \
      --per_device_train_batch_size "1" \
      --gradient_accumulation_steps "32" \
      --learning_rate "4e-6" \
      --warmup_ratio "0.1"  \
      --weight_decay "0.06"  \
      --num_train_epochs "3" \
      --logging_strategy "steps" \
      --logging_steps "100" \
      --evaluation_strategy "steps" \
      --eval_steps  "100" \
      --fp16 "True" \
      --output_dir "/users/local/c20beliv/tmp/$TASK_NAME/" \
      --save_total_limit "1" \
      --ignore_mismatched_sizes "True" > "/users/local/c20beliv/train.log" 2>&1


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
    echo "******* SAVE *******"
    NEW_FILE="/users/local/c20beliv/model/$SAVE_NAME"
    mkdir "/users/local/c20beliv/model/"  # in case it doesn't exit
    mkdir $NEW_FILE

    mv "/users/local/c20beliv/tmp/$TASK_NAME" $NEW_FILE
    mv "/users/local/c20beliv/train.log" $NEW_FILE"/train.log"

    # remove the generated datasets
    echo "remove generated files"
    rm -rf "/users/local/c20beliv/train.log"
    rm -rf $P_FILE"/train_"$SPLIT".json"
    rm -rf $P_FILE"/train_"$SPLIT".mnli.json"

    # remove every thing in the folder containing the model which is not a folder eg we only keep the checkpoints
    # at the end only the last checkpoint remains
    #find //users/local/c20beliv/model/$SAVE_NAME/MNLI/ -type f -delete

  done
done
