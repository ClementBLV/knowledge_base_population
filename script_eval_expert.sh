#!/bin/bash
BASE=$(pwd)

show_help() {
    echo "Usage: source $0 [options]"
    echo
    echo "Options:"
    echo "  --split SPLIT                 Set the split values for the training data, should be an integer in percentage; e.g., 1 means 1% of the training data."
    echo "  --i INDEX                     Set the i values for the training data, should be an integer it is the rank of the training."
    echo "  --model MODEL                 Specify the model; should be a Hugging Face ID, e.g., 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'. It is important to get the right tokenizer"
    echo "  --indirect BOOL               Set the indirect parameter; it's a boolean; if true, it will run the indirect evaluation as well."
    echo "  --saving_name NAME            Specify the save name of the evaluation file."
    echo "  --weights_path PATH           Set the path where the model weights are saved."
    echo "  --parallel BOOL               Boolean to evaluate in parallel."
    echo "  --batch_size INT              Batch size for the evaluation."
    echo "  --processed_test_dir DIR      Set the path where the processed test file for evaluation is stored are saved."
    echo "  --task TASK                   Set the task parameter; it indicates the task, e.g., 'fb' for Freebase or 'wn' for WordNet."
    echo "  --hf_cache_dir DIR              Set the Hugging Face cache directory; if not set, it will be the default directory."
    echo "  --help                        Display this help message and exit."
    exit 0
}

# Parse command-line arguments
declare -A args=(
    [--split]=SPLIT_VALUE
    [--model]=MODEL
    [--indirect]=INDIRECT
    [--saving_name]=SAVE_NAME
    [--weights_path]=WEIGHTS_PATH
    [--parallel]=PARALLEL_BOOL
    [--batch_size]=BATCH_SIZE
    [--processed_test_dir]=P_FILE
    [--task]=TASK
    [--hf_cache_dir]=HF_CACHE_DIR
    [--i]=i
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

# Validate necessary variables
if [ -z "$MODEL" ] || [ -z "$SAVE_NAME" ] || [ -z "$WEIGHTS_PATH" ] || [ -z "$TASK" ]; then
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

# Use the provided parameters
INDIRECT=${INDIRECT:-false}


SAVE_NAME_PREFIX="${SAVE_NAME}"

    
#echo "Split $SPLIT_VALUE v $i"
python3 src/eval.py \
            --input_file $ROOT"preprocessed/test_eval.json" \
            --output_file "${BASE}/eval" \
            --source_model $MODEL \
            --model "${WEIGHTS_PATH}" \
            --saving_name "${SAVE_NAME_PREFIX}" \
            --parallel "${PARALLEL_BOOL}" \
            --batch_size $BATCH_SIZE 

if $INDIRECT; then 
    python3 src/eval.py \
                --input_file "$P_FILE/test_eval_indirect.json" \
                --output_file "${BASE}/eval" \
                --model "${WEIGHTS_PATH}" \
                --name "${SAVE_NAME_PREFIX}_indirect_v${i}" \
                --source_model "$MODEL"
fi