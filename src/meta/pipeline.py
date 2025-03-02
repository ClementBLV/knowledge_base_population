import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Dict, List
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

from trainer_meta import train_model
from eval_meta import evaluate_model
import data2_meta
from src.utils.utils import setup_logger, str2bool, get_config

DATE = datetime.now().strftime("%Y%m%d")


################ setup : parser ################
parser = argparse.ArgumentParser()
parser.add_argument("--train_file", type=str, required=True,
                    help="File with the training mnli data")
parser.add_argument("--test_file", type=str, required=True,
                    help="File with the test mnli data")
parser.add_argument("--valid_file", type=str, required=True,
                    help="File with the valid mnli data")
parser.add_argument("--config_file", type=str, required=True,
                    help="Config file for the meta model")
parser.add_argument("--parallel", type=str2bool, default=False,
                    help="Whether to run the model evaluations in parallel")
parser.add_argument("--train_fraction", type=float, default=0.01,
                    help="Number of examples used for data generation, by default all file will be taken")
parser.add_argument('--output_dir',type=str, required=True,
                help='Directroy to save the outputs (log - weights)')
parser.add_argument("--num_epochs", type=int, default=3,
                    help="Number of epochs")
parser.add_argument("--threshold", type=float, default=0.01,
                    help="Minimum improvement in accuracy required to continue training.")
parser.add_argument("--patience", type=int, default=3,
                    help="Number of iterations to wait for improvement before stopping.")
parser.add_argument('--fast', type=str2bool, default=False,
                    help='Use only 1000 for debug and fast test')
args = parser.parse_args()

################ setup : config ################
config = get_config(args.config_file)

def read_data(input_file : str, fast : bool, logger : logging.Logger):
    with open(input_file) as f:
        datas = json.load(f)
    if fast : 
        logger.warning("DATA : Your are using the FAST mode ! Only 10000 will be taken")
        return datas[0:1000]
    logger.info(f"DATA : {get_total_size(datas)} lines read in the file {input_file}")
    return datas
   
def get_total_size(total_data):
    return len(total_data)


def get_data_fraction(total_data, start_idx=None, end_idx=None):
    if start_idx is not None and end_idx is not None:
        return total_data[start_idx:end_idx]
    else:
        raise ValueError("start_idx and end_idx must be provided.")

    
def save_results(final_results, output_dir, saving_name):
    """Save the final processed results."""
    df = pd.DataFrame(final_results)
    output_file = Path(output_dir) / saving_name
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_file, orient="records", lines=True)
    return output_file


def input2tensor(final_result : List[Dict]): 
    """ Convert the data obtwined with data2_meta.py into usable 
    tensors for training and evaluation"""
    df = pd.DataFrame(final_result)
    y = df['label'].values
    X = []
    filtered_y = []
    for i in range(len(df)):
        l = []
        for p in ['p1', 'p2', 'p3', 'p4']:
            if df.iloc[i][p] is not None:
                ent_indx = config["label2id"]["entailment"]
                l.extend([df.iloc[i][p][0][ent_indx]])
        if l:
            X.append(l)
            filtered_y.append(y[i])
    assert len(X) == len(filtered_y), f"Mismatch: {len(X)} != {len(filtered_y)}"
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(filtered_y, dtype=torch.float32)
    return X_tensor , y_tensor


def df2meta (data_fraction, parallel, config)-> List[Dict]: 
    custom_args = SimpleNamespace(
        datas=data_fraction,
        parallel=parallel,
        config=config
    )
    return data2_meta.main(custom_args)

def pipeline(args):
    output_dir = os.path.join(args.output_dir,f"meta_model_{DATE}")
    os.makedirs( output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(output_dir, "logs.txt"))
    
    print("=========== PIPELINE ============")
    logger.info("Program: pipeline.py ****")
    train_fraction = args.train_fraction
    best_accuracy , best_fraction = 0, 0
    best_model = None

    train_data = read_data(args.train_file, args.fast, logger)
    valid_data = read_data(args.valid_file, args.fast, logger) # --> in the loop 
    test_data = read_data(args.test_file, args.fast, logger) # --> at the end 


    # validation computing 
    valid_data_meta =  df2meta(valid_data, args.parallel, config=config)
    X_tensor_valid, y_tensor_valid = input2tensor(valid_data_meta)

    total_train_size = get_total_size(train_data)

    cached_train_data = [] # Initialize empty training data
    previous_train_size = 0 
    

    while train_fraction <= 1.0:
        step_train_size = int(total_train_size * train_fraction)

        logger.info("******")
        logger.info(f"Data : training step {step_train_size}")

        logger.info(f"Data : Processing {(previous_train_size+step_train_size)/total_train_size * 100:.0f}% of the data...\n")

        ############## training subset ##############
        if previous_train_size + step_train_size <= total_train_size : 
            next_train_size = previous_train_size + step_train_size
            train_data_fraction = get_data_fraction(
                                        train_data, 
                                        start_idx=previous_train_size, 
                                        end_idx=next_train_size)

            previous_train_size = next_train_size 

            cached_train_data.extend(df2meta(train_data_fraction, args.parallel, config=config))
            current_train_fraction = 2*len(cached_train_data)/total_train_size # (2 because reverse and direct)
            
            X_tensor, y_tensor = input2tensor(cached_train_data)

        ############### Train ##############
        model = train_model(X_tensor, y_tensor, num_epochs=args.num_epochs, config_file=args.config_file)
        logger.info(f"Train : training size {len(cached_train_data):.0f}, valid size {len(valid_data):.0f}")

        ############### Evaluate ##############
        accuracy = evaluate_model(model, X_tensor_valid, y_tensor_valid)
        logger.info(f"Metric : Accuracy with {current_train_fraction * 100:.0f}% data: {accuracy:.4f} \n")
        
        ############### Stopping Logic ###############
        if accuracy > best_accuracy + args.threshold:
            best_accuracy = accuracy
            best_fraction = len(cached_train_data)/total_train_size
            best_model = model
            # train_fraction = min(1.0, round(train_fraction + 0.1, 2)) # possibility to also increase the train_fraction for a later version
            no_improve_count = 0  # Reset counter
        else:
            no_improve_count += 1
            logger.info(f"Metric : No significant improvement for {no_improve_count} iterations.")
            logger.info(f"Metric : Best accuracy {best_accuracy:.4f}, Current accuracy {accuracy:.4f} - Treashold {args.threshold}")
            logger.info(f"Metric : Best fraction for training {best_fraction * 100:.0f}% data \n")
            
            if no_improve_count >= args.patience or (current_train_fraction == 1.0 and previous_train_size >= total_train_size):
                logger.info(f"Metric : Stopping training after {no_improve_count} consecutive non-improving iterations.")
                break

    ############### Save ##############
    if best_model:

        # evaluation on test data -- final evaluation of the perf of the meta model 
        test_data_meta =  df2meta(test_data, args.parallel, config=config)
        X_tensor_test, y_tensor_test = input2tensor(test_data_meta)
        last_accuracy = evaluate_model(best_model, X_tensor_test, y_tensor_test)
        logger.info(f"Metric : Accuracy with {current_train_fraction * 100:.0f}% data and Test data: {last_accuracy:.4f} \n")

        # Save the model
        saving_name = f"best_model_{int(last_accuracy * 10000)}.pt"
        model_path = os.path.join(output_dir, saving_name)
        torch.save(best_model.state_dict(), model_path)
        logger.info(f"Data : Best model saved at {model_path} with accuracy {best_accuracy:.4f}")

        # Save the dataset used to the best model
        output_file = save_results(cached_train_data, output_dir, f"training_data_{current_train_fraction * 100}.json")
        logger.info(f"Data : Training set saved to {output_file}")

        # Save the dataset used to the best model
        output_file = save_results(cached_train_data, output_dir, f"valid_data.json")
        logger.info(f"Data : Valid set saved to {output_file}")

        # Save the dataset used for the best model
        output_file = save_results(test_data, output_dir, f"testing_data.json")
        logger.info(f"Data : Testing set saved to {output_file}")

        # Copy the config 
        shutil.copyfile(config["config_path"], os.path.join(output_dir, args.config_file))
   
    return best_model


if __name__ == "__main__":
    pipeline(args)


# TODO : save les logs
