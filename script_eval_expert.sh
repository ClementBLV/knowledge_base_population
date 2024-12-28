#!/bin/bash
BASE=$(pwd)

show_help() {
    echo "Usage: source $0 [options]"
    echo
    echo "Options:"
    echo "  --split SPLIT                 Set the split values for the training data, should be an integer in percentage; e.g., 1 means 1% of the training data."
    echo "  --i INDEX                     Set the i values for the training data, should be an integer it is the rank of the training."
    echo "  --config_file PATH            NAME of the config.json you want to use in the config file."
    echo "  --indirect BOOL               Set the indirect parameter; it's a boolean; if true, it will run the indirect evaluation."
    echo "  --both BOOL                   If true, both direct and indirect evaluations will be performed."
    echo "  --weights_path PATH           Set the path where the model weights are saved."
    echo "  --parallel BOOL               Boolean to evaluate in parallel."
    echo "  --batch_size INT              Batch size for the evaluation."
    echo "  --processed_test_dir DIR      Set the path where the processed test file for evaluation is stored are saved."
    echo "  --task TASK                   Set the task parameter; it indicates the task, e.g., 'fb' for Freebase or 'wn' for WordNet."
    echo "  --hf_cache_dir DIR            Set the Hugging Face cache directory; if not set, it will be the default directory."
    echo "  --fast BOOL                   If set on true, a fast training with only 1000 examples will be done."
    echo "  --help                        Display this help message and exit."
    exit 1
}

# Parse command-line arguments
declare -A args=(
    [--split]=SPLIT_VALUE
    [--config_file]=CONFIG_FILE
    [--indirect]=INDIRECT
    [--both]=BOTH
    [--weights_path]=WEIGHTS_PATH
    [--parallel]=PARALLEL_BOOL
    [--batch_size]=BATCH_SIZE
    [--processed_test_dir]=P_FILE
    [--task]=TASK
    [--hf_cache_dir]=HF_CACHE_DIR
    [--i]=i
    [--fast]=FAST
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
                show_help
            fi
            ;;
    esac
done

# Validate necessary variables
if [ -z "$CONFIG_FILE" ] || [ -z "$WEIGHTS_PATH" ] || [ -z "$TASK" ]; then
    echo "Error: Missing required parameters."
    show_help
fi

if [[ "$TASK" == "wordnet" || "$TASK" == "wn" || "$TASK" == "wn18rr" ]]; then
    ROOT=$BASE"/data/WN18RR/"
else 
    ROOT=$BASE"/data/FB15k237/"
fi

# Hugging Face cache directories
if [ -n "$HF_CACHE_DIR" ] && [ "$HF_CACHE_DIR" != "None" ]; then
    export HF_HOME="$HF_CACHE_DIR/hf/"
    export HF_DATASETS_CACHE="$HF_CACHE_DIR/hf/datasets"
fi

SAVE_NAME_PREFIX="$(basename $WEIGHTS_PATH)"

# Determine settings based on the model name
MODEL_NAME=$(basename "$WEIGHTS_PATH")
DIRECT_EVAL=false
INDIRECT_EVAL=false

if [[ -z "$BOTH_OVERRIDE" && -z "$INDIRECT_OVERRIDE" ]]; then
    if [[ "$MODEL_NAME" == *"2w"* ]]; then
        echo "Model trained for both direct and indirect evaluations (2w)."
        DIRECT_EVAL=true
        INDIRECT_EVAL=true
    elif [[ "$MODEL_NAME" == *"1w"* ]]; then
        if [[ "$MODEL_NAME" == *"direct"* ]]; then
            echo "Model trained for direct evaluation (1w, direct)."
            DIRECT_EVAL=true
        elif [[ "$MODEL_NAME" == *"indirect"* ]]; then
            echo "Model trained for indirect evaluation (1w, indirect)."
            INDIRECT_EVAL=true
        fi
    else
        echo "Error: Unable to determine evaluation settings from model name."
        exit 1
    fi
else
    # User-provided overrides
    DIRECT_EVAL=true  # Default to direct unless overridden
    if [[ "$BOTH_OVERRIDE" == "true" ]]; then
        echo "User override: Running both direct and indirect evaluations."
        INDIRECT_EVAL=true
    elif [[ "$INDIRECT_OVERRIDE" == "true" ]]; then
        echo "User override: Running indirect evaluation."
        INDIRECT_EVAL=true
        DIRECT_EVAL=false
    else
        echo "User override: Running direct evaluation."
        INDIRECT_EVAL=false
    fi
fi


run_evaluation() {
    local direct=$1
    local output_file_suffix=$2
    local save_name=$3
    
    python3 src/data2eval.py \
        --input_file $ROOT"preprocessed/test.json" \
        --direct "$direct" \
        --output_file $ROOT"preprocessed/test_eval_${output_file_suffix}.json" \
        --task "$TASK"

    python3 src/eval.py \
        --input_file $ROOT"preprocessed/test_eval_${output_file_suffix}.json" \
        --output_file "${BASE}/eval" \
        --config_file $CONFIG_FILE \
        --model "${WEIGHTS_PATH}" \
        --saving_name "${save_name}" \
        --parallel "${PARALLEL_BOOL}" \
        --batch_size $BATCH_SIZE \
        --fast $FAST
}

# Run the evaluations based on the determined settings
if $DIRECT_EVAL && $INDIRECT_EVAL; then
    echo "Running both direct and indirect evaluations..."
    run_evaluation true "indirect" "eval_direct_$SAVE_NAME_PREFIX"
    run_evaluation false "direct" "eval_indirect_$SAVE_NAME_PREFIX"
elif $DIRECT_EVAL; then
    echo "Running direct evaluation..."
    run_evaluation false "direct" "eval_$SAVE_NAME_PREFIX"
elif $INDIRECT_EVAL; then
    echo "Running indirect evaluation..."
    run_evaluation true "indirect" "eval_$SAVE_NAME_PREFIX"
else
    echo "Error: No evaluation settings determined."
    exit 1
fi