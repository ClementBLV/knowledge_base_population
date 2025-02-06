
import argparse
from src.combined_model import CombinedModel
from src.utils.utils import get_config


# Paths to your DeBERTa model checkpoints
################ setup : parser ################
parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, required=True,
                    help="File with the training mnli data")
parser.add_argument("--meta_model", type=str, required=True,
                    help="either meta or voting or random")
args = parser.parse_args()


config = get_config(args.config_file)

deberta_paths = [
    config["models_path"]["model_1way"],
    config["models_path"]["model_2ways"],
    config["models_path"]["model_1way_reverse"],
    config["models_path"]["model_1way"]
]

combined_model = CombinedModel(deberta_paths=deberta_paths, meta_model=args.meta_model)
