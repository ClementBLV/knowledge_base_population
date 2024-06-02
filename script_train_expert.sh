#!/bin/bash
BASE=$(pwd)
GLUE_DIR="$BASE/src/"
TASK_NAME="MNLI"
BIAS=true
BOTH=false
SPLIT=7

show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  --splits VALUE                  Set the split values for the training data, should be an integer in percentage; e.g., 1 means 1% of the training data."
    echo "  --both BOOL                     Set the both parameter, it's a boolean; if true, then the direct and indirect relation will be shown; e.g., (true or false)."
    echo "  --bias BOOL                     Set the bias parameter; if true, the sub-set from the training data will be unbalanced as the original one."
    echo "  --task STR                      Set the task parameter; it indicates either the WordNet task or the Freebase one, e.g., --task 'fb' for Freebase or 'wn' for WordNet."
    echo "  --processed_data_directory DIR  Set the directory for processed data in the MNLI format (train_splitted, test, and valid); these files are removed each time."
    echo "  --model MODEL                   Specify the model; should be a Hugging Face ID, e.g., 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'."
    echo "  --output_dir DIR                Set the output directory where the trained weights will be saved."
    echo "  --hf_cache_dir DIR              Set the Hugging Face cache directory; if not set, it will be the default directory."
    echo "                                  If not provided or set to None, the cache settings will not be exported."
    echo "  --help                          Display this help message and exit."
    exit 0
}

# Parse command-line arguments
declare -A args=(
    [--splits]=SPLIT_VALUES
    [--both]=BOTH
    [--bias]=BIAS
    [--task]=TASK
    [--processed_data_directory]=P_FILE
    [--model]=MODEL
    [--output_dir]=OUTPUT_DIR
    [--hf_cache_dir]=HF_CACHE_DIR
)

while [ $# -gt 0 ]; do
    key="$1"
    case $key in
        --help)
            show_help
            ;;
        *)
            if [[ -n "${args[$key]}" ]]; then
                declare "${args[$key]}"="$2"
                shift 2
            else
                echo "Unknown option: $1"
                exit 1
            fi
            ;;
    esac
done

# Hugging Face cache directories
if [ -n "$HF_CACHE_DIR" ] && [ "$HF_CACHE_DIR" != "None" ]; then
    export HF_HOME="$HF_CACHE_DIR/hf/"
    export HF_DATASETS_CACHE="$HF_CACHE_DIR/hf/datasets"
    export TRANSFORMERS_CACHE="$HF_CACHE_DIR/hf/models"
fi

# Validate necessary variables
if [ -z "$P_FILE" ] || [ -z "$MODEL" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required parameters."
    show_help
fi

run_experiment() {
    local i="$1"
    local split="$2"
    local save_name

    case "$MODEL" in
        'microsoft/deberta-v3-base')
            if $BIAS; then
                save_name="untrained_2w_derbertabase_biased_split${split}_v$i"
                echo "UNTRAINED + BIAS!!"
            else
                save_name="untrained_2w_derbertabase_unbiased_split${split}_v$i"
                echo "UNTRAINED + UNBIAS!!"
            fi
            ;;
        'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli')
            if $BIAS; then
                save_name="trained_2w_derbertabase_biased_split${split}_v$i"
                echo "TRAINED + BIAS!!"
            else
                save_name="trained_2w_derbertabase_unbiased_split${split}_v$i"
                echo "TRAINED + UNBIAS!!"
            fi
            ;;
        *)
            echo "Unknown model: $MODEL"
            exit 1
            ;;
    esac

    # Remove old datasets
    echo "Remove old files"
    rm -rf "$P_FILE/train_${split}.mnli.json"

    # Generate a new dataset with new triplets
    source "script.sh" --splits $split --both $BOTH --bias $BIAS --processed_data_directory $P_FILE --task $TASK

    start_time=$(date +%s.%N)
    touch $P_FILE"train.log" # create log file

    echo "=========== TRAIN ============"
    python3 "run_glue.py" \
      --split $split \
      --model_name_or_path "$MODEL" \
      --do_train \
      --do_eval \
      --do_predict \
      --train_file "$P_FILE/train_${split}.mnli.json" \
      --test_file "$P_FILE/test.mnli.json" \
      --validation_file "$P_FILE/valid.mnli.json" \
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
      --output_dir "$OUTPUT_DIR/${TASK_NAME}_${TASK}/$save_name/" \
      --save_total_limit "1" \
      --ignore_mismatched_sizes "True" > "/$P_FILE/train.log" 2>&1

    echo "******* TIME *******"
    end_time=$(date +%s.%N)
    execution_time=$(echo "$end_time - $start_time" | bc)

    hours=$(echo "$execution_time / 3600" | bc)
    minutes=$(echo "($execution_time % 3600) / 60" | bc)
    seconds=$(echo "$execution_time % 60" | bc)

    echo "Execution time: ${hours}h ${minutes}m ${seconds}s"

    # Remove the generated datasets
    echo "Remove generated files"
    rm -rf "$P_FILE/train_${split}.json" "$P_FILE/train_${split}.mnli.json"
}

# Main experiment loop
for i in {1..10}; do
  run_experiment $i $SPLIT
done
