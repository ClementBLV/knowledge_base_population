#!/bin/bash
BASE=$(pwd)
ROOT=$BASE"/data/WN18RR/"
TRAIN=$ROOT"source/train"
TEST=$ROOT"source/test"
VALID=$ROOT"source/valid"

# whoch one to do
TRAIN_BOOL=true
TEST_BOOL=true
VALID_BOOL=true

cd src
# preprocess the raw dataset
python3 data_generator.py --train-path $TRAIN".txt" --valid-path $VALID".txt" --test-path $TEST".txt"



echo "******* TRAIN *******"
# split the train set
#split_values=(5 10 20)

if $TRAIN_BOOL; then
	for SPLIT in 1; do
			# Split train
			echo "$SPLIT"
		!(python3 split.py --input_file $TRAIN".json" --percentage $SPLIT --output_file $ROOT"train_"$SPLIT".json")
		# convert to mnli format
		!(python3 wn2mnli.py --input_file $ROOT"train_"$SPLIT".json" --output_file $ROOT"train_"$SPLIT".mnli.json")
	done
fi

# convert to NLI format
if $TEST_BOOL; then
	echo "******* TEST *******"
	python3 wn2mnli.py --input_file $TEST".json" --output_file $ROOT"test.mnli.json"
fi

if $VALID_BOOL; then
	echo "******* VALID *******"
	python3 wn2mnli.py --input_file $VALID".json" --output_file $ROOT"valid.mnli.json"
fi