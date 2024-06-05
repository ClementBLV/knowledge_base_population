#!/bin/bash

# smaller possible split
split=1
base_dir="$(pwd)/internal_volume"
no_training=false

function usage(){
    echo "entrypoint.sh
    [-s|--split {1..100}]
    [-b|--base_dir <directory to store artifacts]
    [-n|--no_training Whether to train or not]
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
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

if [[ ! -d "${base_dir}" ]]; then
    mkdir -p "${base_dir}"
fi

source ./script_train_expert.sh \
    --split "${split}" \
    --both false \
    --bias true \
    --processed_data_directory "${base_dir}/data/FB15k237/" \
    --model "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli" \
    --output_dir "${base_dir}/weights/FB15k237/" \
    --task "fb" \
    --no_training "${no_training}"
