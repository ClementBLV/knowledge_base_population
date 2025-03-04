#!/bin/bash
BASE=$(pwd)

# which one to do
TRAIN_BOOL=true
TEST_BOOL=true
VALID_BOOL=true
DO_PREPROCESS=true

show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  --split SPLIT_VALUES              Set the split values for the training data, should be an integer in percentage; e.g., 1 means 1% of the training data."
    echo "  --both BOOL                       Set the both parameter, it's a boolean; if true, then the direct and indirect relation will be shown; e.g., (true or false)."
    echo "  --direct BOOL                     Set the direct parameter; if true include only direct relations else include inderect - both parameter overwrite it.  "
    echo "  --bias BOOL                       Set the bias parameter; if true, the sub-set from the training data will be unbalanced as the original one."
    echo "  --task STR                        Set the task parameter; it indicates either the WordNet task or the Freebase one, e.g., --task 'fb' for Freebase or 'wn' for WordNet."
    echo "  --processed_data_directory DIR    Set the directory for processed data in the MNLI format (train_splitted, test, and valid); these files are removed each time."
    echo "  --train_bool BOOL                 Set the train_bool parameter; if true, training will be performed."
    echo "  --test_bool BOOL                  Set the test_bool parameter; if true, testing will be performed."
    echo "  --valid_bool BOOL                 Set the valid_bool parameter; if true, validation will be performed."
    echo "  --do_preprocess BOOL              Set the do_preprocess parameter; if true, preprocessing steps will be executed."
    echo "  --config_file PATH                NAME of the config.json you want to use in the config file"    
    echo "  --help                            Display this help message and exit."
    exit 0
}

# Parse command-line arguments
declare -A args=(
    [--split]=SPLIT_VALUES
    [--direct]=DIRECT
    [--both]=BOTH
    [--bias]=BIAS
    [--task]=TASK
    [--processed_data_directory]=P_FILE
    [--train_bool]=TRAIN_BOOL
    [--test_bool]=TEST_BOOL
    [--valid_bool]=VALID_BOOL
    [--do_preprocess]=DO_PREPROCESS
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

if [[ "$TASK" == "wordnet" || "$TASK" == "wn" || "$TASK" == "wn18rr" ]]; then
    ROOT=$BASE"/data/WN18RR/"
else 
    ROOT=$BASE"/data/FB15k237/"
fi

TRAIN=$ROOT"preprocessed/train"
TEST=$ROOT"preprocessed/test"
VALID=$ROOT"preprocessed/valid"

if [ "$DO_PREPROCESS" == "true" ]; then
    # preprocess the raw dataset
    python3 src/dataprocess/data_generator.py \
                --task "$TASK" \
                --train_path $ROOT"source/train.txt" \
                --valid_path $ROOT"source/valid.txt" \
                --test_path $ROOT"source/test.txt"                             
else
    echo "WARNING : You must have already preprocessed the data" 
fi


# Path 

if [ "$TRAIN_BOOL" == "true" ]; then
    for SPLIT in $SPLIT_VALUES; do 
        # Split train
        python3 src/dataprocess/split.py \
            --input_file $TRAIN".json" \
            --percentage $SPLIT \
            --bias $BIAS\
            --threshold_effectif "923"\
            --output_file $P_FILE"train_"$SPLIT".json"

        # convert to mnli format
        python3 src/dataprocess/data2mnli.py \
            --input_file $P_FILE"train_"$SPLIT".json" \
            --data_source $ROOT \
            --output_file $P_FILE"train_"$SPLIT".mnli.json"\
            --direct $DIRECT \
            --both $BOTH \
            --task "$TASK" \
            --config_name $CONFIG_FILE

        rm -rf $P_FILE"train_"$SPLIT".json"
        echo "INFO : Save : tmp file $P_FILE"train_"$SPLIT".json" was successfully removed"
    done
fi

# convert to NLI format
if [ "$TEST_BOOL" == "true" ]; then
    echo "******* TEST *******"
    python3 src/dataprocess/data2mnli.py \
        --input_file $TEST".json" \
        --data_source $ROOT \
        --output_file $P_FILE"test.mnli.json" \
        --direct $DIRECT \
        --both $BOTH \
        --task "$TASK"\
        --config_name $CONFIG_FILE
fi

if [ "$VALID_BOOL" == "true" ]; then
    echo "******* VALID *******"
    python3 src/dataprocess/data2mnli.py \
        --input_file $VALID".json" \
        --data_source $ROOT \
        --output_file $P_FILE"valid.mnli.json" \
        --direct $DIRECT \
        --both $BOTH \
        --task "$TASK"\
        --config_name $CONFIG_FILE
fi
