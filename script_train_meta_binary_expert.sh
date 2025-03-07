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
    echo "  --do_preprocess BOOL            Set the do_preprocess parameter; if not precise false it will be set on true, preprocessing steps will be executed."    
    echo "  --processed_data_directory DIR  Set the directory for processed data in the MNLI format (train_splitted, test, and valid); these files are removed each time."
    echo "  --saving_name STR               Set the name under which the training data will be saved."
    echo "  --no_training BOOL              If set to true, training will be skipped and 'HELLO WORLD NO TRAINING!' will be printed."
    echo "  --parallel BOOL                 Boolean to evaluate in parallel."
    echo "  --config_file PATH              NAME of the config.json you want to use in the config file"
    echo "  --train_fraction FLOAT          Fraction used for training and testing"
    echo "  --fast BOOL                     If set on true a fast training with only 1000 example will be done"
    echo "  --custom_meta_name STR          Custom name of the meta model" 
    echo "  --output_dir DIR                Custom output dir if not proceised will be the same as processed_data_directory"
    echo "  --help                          Display this help message and exit."
    exit 1
}

# TODO remove the input file it is useless 

# Parse command-line arguments
declare -A args=(
    [--wandb]=WANDB
    [--task]=TASK
    [--input_file]=INPUT_FILE
    [--do_preprocess]=DO_PREPROCESS
    [--processed_data_directory]=P_FILE
    [--saving_name]=SAVING_NAME
    [--no_training]=NO_TRAINING
    [--parallel]=PARALLEL_BOOL
    [--config_file]=CONFIG_FILE
    [--train_fraction]=TRAIN_FRACTION
    [--fast]=FAST
    [--custom_meta_name]=CUSTOM_META_NAME
    [--output_dir]=OUTPUT_DIR
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
if [ -z "$P_FILE" ] ; then
    echo "Error: Missing required parameters."
    show_help
fi

if [[ "$TASK" == "wordnet" || "$TASK" == "wn" || "$TASK" == "wn18rr" ]]; then
    ROOT=$BASE"/data/WN18RR/"
else 
    ROOT=$BASE"/data/FB15k237/"
fi

if [ -z "$DO_PREPROCESS" ]; then
    DO_PREPROCESS=true
fi

if [ -z "$FAST" ]; then
    FAST=false
fi

if [ -z "OUTPUT_DIR"]; then
    OUTPUT_DIR=$P_FILE
fi

run_experiment() {
    # TODO --> mettre le data2mnli avec le train is pas de do_preprocess et le save a cet endroit
    if [ "$DO_PREPROCESS" == "true" ]; then  
        python3 src/dataprocess/data2mnli.py \
            --input_file $ROOT"/preprocessed/train.json" \
            --data_source $ROOT \
            --output_file $P_FILE"/train.mnli.json" \
            --both "True" \
            --task "$TASK" \
            --config_name $CONFIG_FILE \

        python3 src/dataprocess/data2mnli.py \
            --input_file $ROOT"/preprocessed/test.json" \
            --data_source $ROOT \
            --output_file $P_FILE"/test.mnli.json" \
            --both "True" \
            --task "$TASK" \
            --config_name $CONFIG_FILE \

        python3 src/dataprocess/data2mnli.py \
            --input_file $ROOT"/preprocessed/valid.json" \
            --data_source $ROOT \
            --output_file $P_FILE"/valid.mnli.json" \
            --both "True" \
            --task "$TASK" \
            --config_name $CONFIG_FILE \

        #python3 src/data2meta.py  \
        #    --input_file $P_FILE"/$TRAINING_FILE" \
        #    --output_folder $P_FILE \
        #    --saving_name $SAVING_NAME \
        #    --config_file $CONFIG_FILE \
        #    --parallel $PARALLEL_BOOL \
        #    --training_number 10
    fi

    if [ "$NO_TRAINING" == "true" ]; then
        echo "HELLO WORLD NO TRAINING!"
    else
        if [ -n "$CUSTOM_META_NAME" ]; then
            python3 src/meta/pipeline.py   --train_file $P_FILE"/train.mnli.json" \
                                          --test_file $P_FILE"/test.mnli.json" \
                                          --valid_file $P_FILE"/valid.mnli.json" \
                                          --output_dir $OUTPUT_DIR \
                                          --num_epochs 3 \
                                          --config_file $CONFIG_FILE \
                                          --parallel $PARALLEL_BOOL \
                                          --train_fraction $TRAIN_FRACTION \
                                          --fast $FAST \
                                          --custom_meta_name $CUSTOM_META_NAME
        else
            python3 src/meta/pipeline.py   --train_file $P_FILE"/train.mnli.json" \
                                          --test_file $P_FILE"/test.mnli.json" \
                                          --valid_file $P_FILE"/valid.mnli.json" \
                                          --output_dir $OUTPUT_DIR \
                                          --num_epochs 3 \
                                          --config_file $CONFIG_FILE \
                                          --parallel $PARALLEL_BOOL \
                                          --train_fraction $TRAIN_FRACTION \
                                          --fast $FAST
        fi
        # TODO add an argument there for the model output
        
        #echo "=========== TRAIN ============"
        #python3 src/trainer_meta.py --input_file "$P_FILE/$SAVING_NAME" \
        #                            --output_dir $WEIGHT_DIR \
        #                            --num_epochs 3\
        #                            --config_file $CONFIG_FILE 
        
    fi
}

run_experiment
