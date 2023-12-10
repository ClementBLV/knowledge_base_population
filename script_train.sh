#!/bin/bash
BASE=$(pwd)
GLUE_DIR=$BASE"/src/"
TASK_NAME="MNLI"

# wandb
MY_KEY="d454d24f48a094d782b4f38707b2134a3dee9c40"
MY_PROJECT_NAME="knowledge-base"

# Models possible 
MODEL="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
#MODEL="microsoft/deberta-v3-base" 

for i in 6; do
  for SPLIT in 20; do
    ## TO MODIFY 
    SAVE_NAME="trained_derbertabase_biased_split"$SPLIT"_v$i"

    # PATH of the data 
    #P_FILE=$BASE"/data/WN18RR"
    P_FILE="/users/local/c20beliv/"

    ## wandb directory 
    export WANDB_DIR="/users/local/c20beliv/"
    # Set the run name using the WandB command line tool
    export WANDB_RUN_NAME="Deberta-mnli-$SPLIT-v$i"
    
    # remove the old datasets 
    echo "remove old files" 
    rm $P_FILE"/train_"$SPLIT".json"
    rm $P_FILE"/train_"$SPLIT".mnli.json"

    # generate a new dataset which will contain new triplets 
    sh "script.sh" --splits $SPLIT

    start_time=$(date +%s.%N)
    touch "/users/local/c20beliv/train.log" #create log file
    echo "=========== TRAIN ============"

    WANDB_API_KEY=$MY_KEY WANDB_PROJECT=$MY_PROJECT_NAME python3 "src/run_glue.py" \
      --split $SPLIT\
      --model_name_or_path "$MODEL" \
      --do_train \
      --do_eval \
      --do_predict \
      --train_file $P_FILE"/train_$SPLIT.mnli.json" \
      --test_file $BASE"/data/WN18RR/test.mnli.json" \
      --validation_file $BASE"/data/WN18RR/valid.mnli.json" \
      --cache_dir "False" \
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
    rm "/users/local/c20beliv/train.log"
    rm $P_FILE"/train_"$SPLIT".json"
    rm $P_FILE"/train_"$SPLIT".mnli.json"

    # remove every thing in the folder containing the model which is not a folder eg we only keep the checkpoints
    # at the end only the last checkpoint remains  
    find //users/local/c20beliv/model/$SAVE_NAME/MNLI/ -type f -delete

  done
done
