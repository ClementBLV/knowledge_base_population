#!/bin/bash
BASE=$(pwd)


# whoch one to do
TRAIN_BOOL=true
TEST_BOOL=true
VALID_BOOL=true

# Parse command-line arguments
declare -A args=(
    [--splits]=SPLIT_VALUES
    [--both]=BOTH
    [--bias]=BIAS
    [--task]=TASK
    [--processed_data_directory]=P_FILE
)

while [ $# -gt 0 ]; do
    key="$1"
    if [[ -n "${args[$key]}" ]]; then
        declare "${args[$key]}"="$2"
        shift 2
    else
        echo "Unknown option: $1"
        exit 1
    fi
done


if [[ "$TASK" == "wordnet" || "$TASK" == "wn" || "$TASK" == "wn18rr" ]]; then
    ROOT=$BASE"/data/WN18RR/"
else 
    ROOT=$BASE"/data/FB15k237/"
fi

TRAIN=$ROOT"source/train"
TEST=$ROOT"source/test"
VALID=$ROOT"source/valid"

cd src
# preprocess the raw dataset
python3 data_generator.py \
            --task "$TASK" \
            --train-path $TRAIN".txt" \
            --valid-path $VALID".txt" \
            --test-path $TEST".txt"



# Path 

if $TRAIN_BOOL; then
    for SPLIT in $SPLIT_VALUES ; do 
		# Split train

		echo "split $SPLIT %"

		python3 split.py \
			--input_file $TEST".json" \
			--percentage $SPLIT \
			--bias $BIAS\
			--threshold_effectif "923"\
			--output_file $P_FILE"train_"$SPLIT".json"

		# convert to mnli format
		python3 wn2mnli.py \
			--input_file $P_FILE"train_"$SPLIT".json" \
			--output_file $P_FILE"train_"$SPLIT".mnli.json"\
			--both $BOTH \
            --task "$TASK"""

		rm -rf $P_FILE"train_"$SPLIT".json"
        echo "tmp file $P_FILE"train_"$SPLIT".json" was successfully removed"

	done
fi

# convert to NLI format
if $TEST_BOOL; then
	echo "******* TEST *******"
	python3 wn2mnli.py \
        --input_file $TEST".json" \
        --output_file $ROOT"test.mnli.json" \
        --both $BOTH \
        --task "$TASK"""
fi

if $VALID_BOOL; then
	echo "******* VALID *******"
	python3 wn2mnli.py \
        --input_file $VALID".json" \
        --output_file $ROOT"valid.mnli.json" \
        --both $BOTH \
        --task "$TASK"""
fi