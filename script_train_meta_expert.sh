#!/bin/bash
BASE=$(pwd)
TASK_NAME="MNLI"


show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  --wandb BOOL                    Set the wandb parameter; if true, wandb will be called."
    echo "  --task STR                      Set the task parameter; it indicates either the WordNet task or the Freebase one, e.g., --task 'fb' for Freebase or 'wn' for WordNet."
    echo "  --input_file STR                Set the input file (name of the .json file) of the data to preprocess"
    echo "  --training_file_name STR        Set the name under which the training file will be saved after preprocessing or have been saved"
    echo "  --do_preprocess BOOL            Set the do_preprocess parameter; if not precise false it will be set on true, preprocessing steps will be executed."    
    echo "  --processed_data_directory DIR  Set the directory for processed data in the MNLI format (train_splitted, test, and valid); these files are removed each time."
    echo "  --weight_dir DIR                Set the output directory where the trained weights will be saved."
    echo "  --saving_name STR               Set the name under which the training data will be saved."
    echo "  --no_training BOOL              If set to true, training will be skipped and 'HELLO WORLD NO TRAINING!' will be printed."
    echo "  --parallel BOOL                 Boolean to evaluate in parallel."
    echo "  --config_file PATH              NAME of the config.json you want to use in the config file"
    echo "  --help                          Display this help message and exit."
    exit 0
}

# Parse command-line arguments
declare -A args=(
    [--wandb]=WANDB
    [--task]=TASK
    [--input_file]=INPUT_FILE
    [--training_file_name]=TRAINING_FILE
    [--do_preprocess]=DO_PREPROCESS
    [--processed_data_directory]=P_FILE
    [--weight_dir]=WEIGHT_DIR
    [--saving_name]=SAVING_NAME
    [--no_training]=NO_TRAINING
    [--parallel]=PARALLEL_BOOL
    [--config_file]=CONFIG_FILE

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
if [ -z "$P_FILE" ] || [ -z "$WEIGHT_DIR" ]; then
    echo "Error: Missing required parameters."
    show_help
fi

if [[ "$TASK" == "wordnet" || "$TASK" == "wn" || "$TASK" == "wn18rr" ]]; then
    ROOT=$BASE"/data/WN18RR/preprocessed/"
else 
    ROOT=$BASE"/data/FB15k237/preprocessed"
fi

if [ -z "$DO_PREPROCESS" ]; then
    DO_PREPROCESS=true
fi


run_experiment() {
    # TODO --> mettre le data2mnli avec le train is pas de do_preprocess et le save a cet endroit
    if [ "$DO_PREPROCESS" == "true" ]; then  

        python3 src/data2meta_v2.py  \
            --input_file $ROOT"/$INPUT_FILE" \
            --output_folder $P_FILE \
            --saving_name $SAVING_NAME \
            --config_file $CONFIG_FILE \
            --parallel $PARALLEL_BOOL \
            --training_number 10 \
            --task $TASK
    fi

    if [ "$NO_TRAINING" == "true" ]; then
        echo "HELLO WORLD NO TRAINING!"
    else
        # create log file

        echo "=========== TRAIN ============"
        python3 src/trainer_meta.py --input_file "$P_FILE/$SAVING_NAME" \
                                    --output_dir $WEIGHT_DIR \
                                    --num_epochs 3\
        
    fi
}

run_experiment
