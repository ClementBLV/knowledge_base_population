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

# Parse command-line arguments
while [ $# -gt 0 ]; do
    key="$1"

    case $key in
        --splits)
        SPLIT_VALUES="$2"
        shift # past argument
        shift # past value
        ;;
        --both)
        BOTH="$2"
        shift # past argument
        shift # past value
        ;;
		--bias)
        BIAS="$2"
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
python3 data_generator.py --task "wn" --train-path $TRAIN".txt" --valid-path $VALID".txt" --test-path $TEST".txt"

echo $BIAS
# Path 
#P_FILE=$ROOT
P_FILE="/users/local/c20beliv/"

#--threshold_effectif \
if $TRAIN_BOOL; then
    for SPLIT in $SPLIT_VALUES ; do 
		# Split train
        echo $BIAS
		echo "split $SPLIT %"
		!(python3 split.py \
			--input_file $TRAIN".json" \
			--percentage $SPLIT \
			--bias $BIAS\
			--threshold_effectif "923"\
			--output_file $P_FILE"train_"$SPLIT".json")
		# convert to mnli format
		!(python3 wn2mnli.py \
			--input_file $P_FILE"train_"$SPLIT".json" \
			--output_file $P_FILE"train_"$SPLIT".mnli.json"\
			--both $BOTH)
		rm -rf $P_FILE"train_"$SPLIT".json"
	done
fi

# convert to NLI format
if $TEST_BOOL; then
	echo "******* TEST *******"
	python3 wn2mnli.py --input_file $TEST".json" --output_file $ROOT"test.mnli.json" --both $BOTH
fi

if $VALID_BOOL; then
	echo "******* VALID *******"
	python3 wn2mnli.py --input_file $VALID".json" --output_file $ROOT"valid.mnli.json" --both $BOTH
fi