#!/bin/bash
BASE=$(pwd)
ROOT=$BASE"/data/WN18RR/"
TRAIN=$ROOT"source/train"
TEST=$ROOT"source/test"
VALID=$ROOT"source/valid"

# whoch one to do
TRAIN_BOOL=true
TEST_BOOL=false
VALID_BOOL=false

# Parse command-line arguments
while [ $# -gt 0 ]; do
    key="$1"

    case $key in
        --splits)
        SPLIT_VALUES="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

cd src
# preprocess the raw dataset
python3 data_generator.py --train-path $TRAIN".txt" --valid-path $VALID".txt" --test-path $TEST".txt"

# Path 
#P_FILE=$ROOT
P_FILE="/users/local/c20beliv/"


if $TRAIN_BOOL; then
	#IFS=', ' read -r -a SPLIT_ARRAY <<< "$SPLIT_VALUES"
    for SPLIT in $SPLIT_VALUES ; do #"${SPLIT_ARRAY[@]}"
		# Split train
		echo "split $SPLIT %"
		!(python3 split.py \
			--input_file $TRAIN".json" \
			--percentage $SPLIT \
			--bias "True"\
			--threshold_effectif "923"\
			--output_file $P_FILE"train_"$SPLIT".json")
		# convert to mnli format
		!(python3 wn2mnli.py --input_file $P_FILE"train_"$SPLIT".json" --output_file $P_FILE"train_"$SPLIT".mnli.json")
		rm -rf $P_FILE"train_"$SPLIT".json"
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