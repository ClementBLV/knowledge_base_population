#!/bin/bash
BASE=$(pwd)
GLUE_DIR="$BASE/src/"
TASK_NAME="MNLI"
#BIAS=true
#BOTH=false

show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  --split SPLIT                   Set the split values for the training data, should be an integer in percentage; e.g., 1 means 1% of the training data."
    echo "  --direct BOOL                   Set the direct parameter; if true include only direct relations else include inderect - both parameter overwrite it.  "
    echo "  --both BOOL                     Set the both parameter, it's a boolean; if true, then the direct and indirect relation will be shown; e.g., (true or false)."
    echo "  --bias BOOL                     Set the bias parameter; if true, the sub-set from the training data will be unbalanced as the original one."
    echo "  --wandb BOOL                    Set the wandb parameter; if true, wandb will be called."
    echo "  --task STR                      Set the task parameter; it indicates either the WordNet task or the Freebase one, e.g., --task 'fb' for Freebase or 'wn' for WordNet."
    echo "  --processed_data_directory DIR  Set the directory for processed data in the MNLI format (train_splitted, test, and valid); these files are removed each time."
    echo "  --do_preprocess BOOL            Set the do_preprocess parameter; if not precise false it will be set on true, preprocessing steps will be executed."
    echo "  --output_dir DIR                Set the output directory where the trained weights will be saved."
    echo "  --hf_cache_dir DIR              Set the Hugging Face cache directory; if not set, it will be the default directory."
    echo "                                  If not provided or set to None, the cache settings will not be exported."
    echo "  --no_training BOOL              If set to true, training will be skipped and 'HELLO WORLD NO TRAINING!' will be printed."
    echo "  --config_file PATH              NAME of the config.json you want to use in the config file"
    echo "  --custom_name STR               Custom name to save the model"
    echo "  --wandb_key STR                 The API key for wandb logging"
    echo "  --fast BOOL                     If set on true a fast training with only 1000 example will be done"
    echo "  --help                          Display this help message and exit."
    exit 0
}

# Parse command-line arguments
declare -A args=(
    [--split]=SPLIT_VALUE
    [--direct]=DIRECT
    [--both]=BOTH
    [--bias]=BIAS
    [--wandb]=WANDB
    [--task]=TASK
    [--processed_data_directory]=P_FILE
    [--do_preprocess]=DO_PREPROCESS
    [--output_dir]=OUTPUT_DIR
    [--hf_cache_dir]=HF_CACHE_DIR
    [--no_training]=NO_TRAINING
    [--config_file]=CONFIG_FILE
    [--custom_name]=CUSTOM_NAME
    [--wandb_key]=WANK_KEY
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
                exit 0
            fi
            ;;
    esac
done

# Hugging Face cache directories
if [ -n "$HF_CACHE_DIR" ] && [ "$HF_CACHE_DIR" != "None" ]; then
    export HF_HOME="$HF_CACHE_DIR/hf/"
    export HF_DATASETS_CACHE="$HF_CACHE_DIR/hf/datasets"
    #export TRANSFORMERS_CACHE="$HF_CACHE_DIR/hf/models"
fi

# Validate necessary variables
if [ -z "$P_FILE" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$CONFIG_FILE" ]; then
    echo "Error: Missing required parameters."
    show_help
fi

if [ -z "$DO_PREPROCESS" ]; then
    DO_PREPROCESS=true
fi

run_experiment() {
    local i="$1"
    local split="$2"
    local save_name
    local DO_PREPROCESS="$3"  # Optionally pass this from outside

    echo "=========== iteration $i ============"
    # Check if i > 1 or DO_PREPROCESS has already been set to false by the user
    if [ "$i" -gt 1 ] || [ "$DO_PREPROCESS" == "false" ]; then
        TEST_BOOL=false
        VALID_BOOL=false
        DO_PREPROCESS=false
    else
        TEST_BOOL=true
        VALID_BOOL=true
        DO_PREPROCESS=true
    fi
    
    save_name=$(python3 src/name_generation.py "$CONFIG_FILE" "$BIAS" "$DIRECT" "$BOTH" "$SPLIT" "$VERSION" "$CUSTOM_NAME")

    # Remove old datasets
    echo "Remove old files"
    rm -rf "$P_FILE/train_${split}.mnli.json"
    # Generate a new dataset with new triplets
    source "script.sh" \
     --split $split \
     --direct $DIRECT \
     --both $BOTH \
     --bias $BIAS \
     --processed_data_directory $P_FILE \
     --task $TASK \
     --test_bool $TEST_BOOL \
     --valid_bool $VALID_BOOL \
     --do_preprocess $DO_PREPROCESS \
     --config_file $CONFIG_FILE

    if [ "$NO_TRAINING" == "true" ]; then
        echo "HELLO WORLD NO TRAINING!"
    else
        # create log file
        touch "$BASE/log/train_${save_name}.log" 

        echo "=========== TRAIN ============"
        python3 "src/trainer_hf.py" \
            --do_train "yes"\
            --train_file "${P_FILE}train_${split}.mnli.json" \
            --test_file "${P_FILE}test.mnli.json" \
            --output_dir "$OUTPUT_DIR/${TASK_NAME}_${TASK}/" \
            --save_name $save_name \
            --config_file $CONFIG_FILE \
            --wandb_api_key $WANK_KEY \
            --fast $FAST
    fi

    # Remove the generated datasets
    echo "=========== CLEANING ============"
    echo -e "Remove generated files : \n $P_FILE/train_${split}.json \n $P_FILE/train_${split}.mnli.json"
    rm -rf "$P_FILE/train_${split}.json" "$P_FILE/train_${split}.mnli.json"
}


# Check if SPLIT_VALUE is 100 and run experiment only once if true (no variance in that case)
if [ "$SPLIT_VALUE" -eq 100 ]; then
    echo "WARNING : you are running the experiment with a split = 100 % "
    echo "          It will be launched only ones as no variance is induced "
    echo "          as no sampling is done on the dataset "

    run_experiment 1 $SPLIT_VALUE
else
    # Main experiment loop
    for i in {1..1}; do
        run_experiment 1 $SPLIT_VALUE $DO_PREPROCESS
    done
fi
