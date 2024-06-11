#!/bin/bash

# smaller possible split
split=1
base_dir="$(pwd)/internal_volume"
no_training=false
both=false
model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

function usage(){
    echo "entrypoint.sh
    [-s|--split {1..100}]
    [-b|--base_dir <directory to store artifacts]
    [-n|--no_training Whether to train or not]
    [-t|--both Whether to set --both to true or not]
    [-m|--model_name <model name>]
"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--split)
            split="$2"
            shift; shift
            ;;
        -b|--base_dir)
            base_dir="$2"
            shift; shift
            ;;
        -n|--no_training)
            no_training=true
            shift
            ;;
        -t|--both)
            both=true
            shift
            ;;
        -m|--model_name)
            model_name="$2"
            shift; shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

if [[ ! -d "${base_dir}" ]]; then
    mkdir -p "${base_dir}"
fi

source ./script_train_expert.sh \
    --split "${split}" \
    --both "${both}" \
    --bias true \
    --processed_data_directory "${base_dir}/data/FB15k237/" \
    --model "${model_name}" \
    --output_dir "${base_dir}/weights/FB15k237/" \
    --task "fb" \
    --no_training "${no_training}"
