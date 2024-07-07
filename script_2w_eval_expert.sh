#!/bin/bash

# Default values for variables
BASE=$(pwd)
INPUT_FILE="${BASE}/data/FB15k237/test_eval.json"
INPUT_FILE_INDIRECT="${BASE}/data/FB15k237/test_eval_indirect.json"
OUTPUT_FILE="${BASE}/eval"
NORMALISE="true"

show_help() {
    echo "Usage: source $0 [options]"
    echo
    echo "Options:"
    echo "  --weights_path MODEL          Specify the model; should be the path to inside a checkpoints"
    echo "  --processed_data_directory DIR      Set the path where the processed test file for evaluation is stored are saved (here indirect and direct test set)."
    echo "  --input_file FILE             Path to the input file (default: ${BASE}/data/FB15k237/test_eval.json)"
    echo "  --input_file_indirect FILE    Path to the indirect input file (default: ${BASE}/data/FB15k237/test_eval_indirect.json)"
    echo "  --output_file FILE            Path to the output file (default: ${BASE}/eval)"
    echo "  --model_name NAME             Model name for naming convention (could be Naive or MNLI for eg)"
    echo "  --normalise BOOL              Normalisation flag (default: true)"
    echo "  --help                        Display this help message and exit."
    exit 0
}

# Parse command-line arguments
declare -A args=(
    [--weights_path]=WEIGHTS_PATH
    [--processed_data_directory]=P_FILE
    [--output_file]=OUTPUT_FILE
    [--model_name]=MODEL_NAME
    [--normalise]=NORMALISE
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

echo "******* EVAL *******"


# creation of the test files : 
# generate test eval file only once
python3 src/wn2eval.py \
    --input_file $TEST".json" \
    --output_file "$P_FILE/test_eval_indirect.json" \
    --task "$TASK" \
    --direct false \

# evaluation : 
python3 eval_2w_final.py \
    --input_file  "$P_FILE/test_eval.json" \
    --input_file_indirect  "$P_FILE/test_eval_indirect.json"    \
    --output_file $OUTPUT_FILE \
    --model $WEIGHTS_PATH \
    --name "2way_eval_${MODEL_NAME}_100" \
    --source_model $MODEL \
    --normalise $NORMALISE
