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
import json
import random
import sys
import numpy as np
import pandas as pd
import torch 
import gc
from accelerate.utils import release_memory
import shutil
import wandb
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer, TrainerCallback

from datasets import load_dataset

from datetime import datetime

SEED_GLOBAL = 42
DATE =  datetime.today().strftime("%Y%m%d")
FAST = False



################ setup : logger ################
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.info("Progam : hf_trainer.py ****")

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
parser.add_argument("--config_file", type=str, required=True, 
                    help="Nale if the config file of for the meta model")
parser.add_argument('--wandb_api_key', type=str, required=True,
                    help='Weights & Biases API key for logging')
args = parser.parse_args()

################ setup : config ################
current_dir = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(current_dir), "configs", args.config_file)
with open(config_path, "r") as config_file:
    config = json.load(config_file)

################ setup : dirrectories ################
os.makedirs(args.output_dir, exist_ok=True)
logging_directory = os.path.join(args.output_dir, "logs")
os.makedirs(logging_directory, exist_ok=True)
logger.info(f"Save : Checkpoint Save location : {args.output_dir}")
logger.info(f"Save : Trained model saving Name : {args.model_name}")

################ load : model ################
model_name = args.model_name
max_length = config["max_length"]
assert (config["model_name"]==model_name) ,"The config isn't the one of the model used"

# label2id mapping
if args.do_train:
    label2id = config["label2id"]#{"entailment": 0, "not_entailment": 1}  #{"entailment": 0, "neutral": 1, "contradiction": 2}
    id2label = config["id2label"]#{0: "entailment", 1: "not_entailment"}  #{0: "entailment", 1: "neutral", 2: "contradiction"}
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

############### setup : wandb ################

if args.do_train :
    logger.info("Setting up wandb for logging...")
    wandb.login(key=args.wandb_api_key)  # Use the API key from args

    # Initialize wandb run
    wandb.init(
        project="your_project_name",  # Replace with your wandb project name
        config={
            "learning_rate": config["learning_rate"],
            "epochs": config["num_train_epochs"],
            "batch_size": config["per_device_train_batch_size"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "model_name": model_name,
            "seed": SEED_GLOBAL,
        },
        name=f"run_{args.save_name}_{DATE}",
    )
    logger.info("wandb initialized.")

################ load : data ################
raw_datasets = load_dataset(
                "json",
                data_files={
                                "train": args.train_file,
                                "test": args.test_file,
                            },
                #cache_dir=model_args.cache_dir,
                #token=model_args.token,
            )

dataset_train = raw_datasets["train"]
dataset_test  = raw_datasets["test"]
if FAST : 
    dataset_train = dataset_train.select(range(1000))
    dataset_test = dataset_test.select(range(500))


################
# TOKENIZER
################


def tokenize_func_naive(examples):
    """ without padding="max_length" & max_length=512, it should do dynamic padding."""
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)  # max_length=512,  padding=True

# Parameters
max_length_chars = int(max_length * 0.6)  # Approximate character count threshold
skip_counter , total_pairs = 0, 0

# Step 1: Filter the dataset based on length
if "small" in model_name:
    def filter_func(example):
        global skip_counter, total_pairs
        total_pairs += 1
        
        combined_text = example["premise"] + " " + example["hypothesis"]
        if len(combined_text.split()) <= max_length_chars:
            return True
        else:
            skip_counter += 1
            return False

    # Apply filtering
    logger.warning("Prefilter the data to avoid long sequence when using a small model")
    dataset_train = dataset_train.filter(filter_func)
    skipped_percentage = (skip_counter / total_pairs) * 100
    logger.info(f"Train : Skipped {skip_counter} pairs ({skipped_percentage:.2f}%) due to length exceeding {max_length} tokens.")

    skip_counter , total_pairs = 0, 0
    dataset_test = dataset_test.filter(filter_func)
    skipped_percentage = (skip_counter / total_pairs) * 100
    logger.info(f"Test : Skipped {skip_counter} pairs ({skipped_percentage:.2f}%) due to length exceeding {max_length} tokens.")

# Step 2: Tokenize the filtered dataset
def tokenize_func(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation="longest_first",
        max_length=max_length,
        padding="max_length",
    )

# Tokenize the filtered datasets
logger.info("Train : tokenization")
encoded_dataset_train = dataset_train.map(tokenize_func_naive, batched=True)
logger.info("Test : tokenization")
encoded_dataset_test = dataset_test.map(tokenize_func_naive, batched=True)


logger.info(f"len train = {len(encoded_dataset_train)} , len test = {len(encoded_dataset_test)}")
print(encoded_dataset_test.column_names)
# remove columns the library does not expect
encoded_dataset_train = encoded_dataset_train.remove_columns(["hypothesis", "premise"])
encoded_dataset_test = encoded_dataset_test.remove_columns(["hypothesis", "premise"])


################
# TRAINER
################

################ training args ################

fp16_bool = True if torch.cuda.is_available() else False
if "mDeBERTa" in model_name: fp16_bool = False  # mDeBERTa does not support FP16 yet

# https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
eval_batch = config["eval_batch"]#64 if "large" in model_name else 64*2
per_device_train_batch_size = config["per_device_train_batch_size"] #8 if "large" in model_name else 32
gradient_accumulation_steps = config["gradient_accumulation_steps"] #4 if "large" in model_name else 1
warmup_ratio=config["warmup_ratio"]
weight_decay=config["weight_decay"]
num_train_epochs=config["num_train_epochs"]
learning_rate=config["learning_rate"]

train_args = TrainingArguments(
    output_dir=args.output_dir,
    logging_dir=logging_directory,
    #deepspeed="ds_config_zero3.json",  # if using deepspeed
    lr_scheduler_type= config["lr_scheduler_type"] ,
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
    save_strategy=config["save_strategy"],  # options: "no"/"steps"/"epoch"
    save_steps=config["save_steps"],  # Number of updates steps before two checkpoint saves.
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
# Custom callback class to log metrics to Wandb
class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to wandb during training."""
        if wandb.run:
            phase = "train" if state.global_step > 0 else "eval"
            for key, value in logs.items():
                wandb.log({f"{phase}_{key}": value})

def log_metrics(metrics, phase="train"):
    """Log metrics to wandb."""
    if wandb.run:
        for key, value in metrics.items():
            wandb.log({f"{phase}_{key}": value})

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=train_args,
    train_dataset=encoded_dataset_train,  
    eval_dataset=encoded_dataset_test,  
    callbacks=[WandbCallback],  # Use the custom WandbCallback

)
################ train ################
if args.do_train:
    train_result = trainer.train()

    model_path = f"{args.output_dir}/{args.save_name}-{DATE}"
    trainer.save_model(output_dir=model_path)
    shutil.copy(config_path, f"{model_path}/config_used.json")
    
    # Log final metrics
    final_metrics = train_result.metrics
    log_metrics(final_metrics, "final_train")

    # Close wandb run
    if wandb.run:
        wandb.config.update({"model_path": model_path})
        wandb.finish()  
################ save ################


if device == "cuda":
    # free memory
    flush()
    release_memory(model)
    del (model, trainer)
