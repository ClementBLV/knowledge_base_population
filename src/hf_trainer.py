#########################################
####### Code to train the model ######### 
#########################################

################
# IMPORT - SETUP
################

################ import ################
import argparse
import logging
import os
import random
import numpy as np
import pandas as pd
import torch 
import gc
from accelerate.utils import release_memory

import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

from datetime import datetime
import logger

SEED_GLOBAL = 42
DATE =  datetime.today().strftime("%Y%m%d")

################ setup : logger ################
logger = logging.getLogger(__name__)
logger.setup_logging()

################ setup : seed ################
np.random.seed(SEED_GLOBAL)
torch.manual_seed(SEED_GLOBAL)
random.seed(SEED_GLOBAL)

################ setup : cuda ################
device = "cuda" if torch.cuda.is_available() else "cpu"
def flush():
  # release memory: https://huggingface.co/blog/optimize-llm
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()


################ setup : args ################
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Parser for terminal arguments")
parser.add_argument('-train', '--do_train', type=str2bool, default=True,
                    help='Do training of flag set. Otherwise only evaluation')
parser.add_argument('-train_file', '--train_file',type=str, required=True,
                    help='Json file with the processed training data')
parser.add_argument('-test_file', '--test_file',type=str, required=True,
                    help='Json file with the processed testing data')
parser.add_argument('-model_name', '--model_name',type=str, required=True,
                    help='Model name of the model to train, must be an HF ID')
parser.add_argument('-output_dir', '--output_dir',type=str, required=True,
                    help='Directroy to save the outputs (log - weights)')
parser.add_argument('-save_name', '--save_name',type=str, required=True,
                    help='Name under which the model will be saved in the output dir')
args = parser.parse_args()

################ setup : dirrectories ################
os.makedirs(args.output_dir, exist_ok=True)
logging_directory = os.path.join(args.output_dir, "logs")
os.makedirs(logging_directory, exist_ok=True)

################ load : model ################

model_name = args.model_name
max_length = 512

# label2id mapping
if args.do_train:
    label2id = {"entailment": 0, "not_entailment": 1}  #{"entailment": 0, "neutral": 1, "contradiction": 2}
    id2label = {0: "entailment", 1: "not_entailment"}  #{0: "entailment", 1: "neutral", 2: "contradiction"}
    logger.info(f"Label used : {list(label2id.keys())}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=False, 
        model_max_length=max_length
    )  # model_max_length=512
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        label2id=label2id, 
        id2label=id2label
    ).to(device)

    label_text_unique = list(label2id.keys())
    print(label_text_unique)
else:
    logger.error("This script is for training not evaluation")
    raise ValueError("You must use this script for training")


################ load : data ################
raw_datasets = load_dataset(
                "json",
                data_files={
                                "train": args.train_file,
                                "validation": args.test_file,
                            },
                #cache_dir=model_args.cache_dir,
                #token=model_args.token,
            )
dataset_train = raw_datasets["train"]
dataset_test  = raw_datasets["test"]
dataset_train_filtered = dataset_train.select(range(1000))


################
# TOKENIZER
################


def tokenize_func(examples):
    """ without padding="max_length" & max_length=512, it should do dynamic padding."""
    return tokenizer(examples["text"], examples["hypothesis"], truncation=True)  # max_length=512,  padding=True


################ encoded data ################
encoded_dataset_train = dataset_train_filtered.map(tokenize_func, batched=True)
encoded_dataset_test = dataset_test.map(tokenize_func, batched=True)
logger.info(f"len train = {len(encoded_dataset_train)} , len test = {len(encoded_dataset_test)}")

# remove columns the library does not expect
encoded_dataset_train = encoded_dataset_train.remove_columns(["hypothesis", "text"])
encoded_dataset_test = encoded_dataset_test.remove_columns(["hypothesis", "text"])


################
# TRAINER
################

################ training args ################

fp16_bool = True if torch.cuda.is_available() else False
if "mDeBERTa" in model_name: fp16_bool = False  # mDeBERTa does not support FP16 yet

# https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
eval_batch = 64 if "large" in model_name else 64*2
per_device_train_batch_size = 8 if "large" in model_name else 32
gradient_accumulation_steps = 4 if "large" in model_name else 1
warmup_ratio=0.06
weight_decay=0.01
num_train_epochs=3
learning_rate=9e-6

train_args = TrainingArguments(
    output_dir=args.output_dir,
    logging_dir=logging_directory,
    #deepspeed="ds_config_zero3.json",  # if using deepspeed
    lr_scheduler_type= "linear",
    group_by_length=False,  # can increase speed with dynamic padding, by grouping similar length texts https://huggingface.co/transformers/main_classes/trainer.html
    learning_rate=learning_rate if "large" in model_name else 2e-5,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=eval_batch, 
    gradient_accumulation_steps=gradient_accumulation_steps, # (!adapt/halve batch size accordingly). accumulates gradients over X steps, only then backward/update. decreases memory usage, but also slightly speed
    #eval_accumulation_steps=2,
    num_train_epochs=num_train_epochs,
    #max_steps=400,
    #warmup_steps=0,  # 1000,
    warmup_ratio=warmup_ratio, #0.1, 0.06
    weight_decay=weight_decay, #0.1,
    fp16=fp16_bool, # ! only makes sense at batch-size > 8. loads two copies of model weights, which creates overhead. https://huggingface.co/transformers/performance.html?#fp16
    fp16_full_eval=fp16_bool,
    evaluation_strategy="epoch",
    seed=SEED_GLOBAL,
    #load_best_model_at_end=True,
    #metric_for_best_model="accuracy",
    #eval_steps=300,  # evaluate after n steps if evaluation_strategy!='steps'. defaults to logging_steps
    save_strategy="steps",  # options: "no"/"steps"/"epoch"
    save_steps=1000,  # Number of updates steps before two checkpoint saves.
    #save_total_limit=1,  # If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir
    #logging_strategy="epoch",
    #logging_steps=100,
    #report_to="all",  # "all"
    #run_name=run_name,
    #push_to_hub=True,  # does not seem to work if save_strategy="no"
    #hub_model_id=hub_model_id,
    #hub_token=config.HF_ACCESS_TOKEN,
    #hub_strategy="end",
    #hub_private_repo=True,
)

################ trainer ################
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=train_args,
    train_dataset=encoded_dataset_train,  #.shard(index=1, num_shards=200),  # https://huggingface.co/docs/datasets/processing.html#sharding-the-dataset-shard
    eval_dataset=encoded_dataset_test,  #.shard(index=1, num_shards=20),
    #compute_metrics=lambda x: compute_metrics_standard(x, label_text_alphabetical=label_text_unique)  #compute_metrics,
)

################ train ################
if args.do_train:
    trainer.train()


################ save ################
model_path = f"{args.output_dir}/{args.save_name}-{DATE}"
trainer.save_model(output_dir=model_path)

if device == "cuda":
    # free memory
    flush()
    release_memory(model)
    del (model, trainer)