import json
import os
import sys
import warnings
import logging 
import re

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)



def extract_size(sentence):
    match = re.search(r'\b(base|small|large)\b', sentence, re.IGNORECASE)
    if match:
        return match.group(0).lower()
    return "unkown"

def generate_save_name(config_file, bias, direct, both, split, version, custom_name=None):

    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(os.path.dirname(current_dir), "configs", config_file)
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    model = config["model_name"]

    if custom_name:
        return custom_name

    if "microsoft" in model :
        prefix = "naive"
    elif "MoritzLaurer" in model:
        prefix = "mnli"
    else:
        prefix = f"{model.replace('/', '_')}_split{split}_v{version}"
    
    both_indicator = "2w" if both else "1w"
    if not both : 
        direct_indicator = "direct" if direct else "indirect"
        both_indicator = f"{both_indicator}_{direct_indicator}"

    bias_indicator = "biased" if bias else "unbiased"

    size = extract_size(model)

    save_name = f"{prefix}_derberta_{size}_{both_indicator}_{bias_indicator}_split{split}_v{version}"
    return save_name

if __name__ == "__main__":
    config_file = sys.argv[1]
    bias = sys.argv[2].lower() in ["true", "True"]
    direct = sys.argv[3].lower() in ["true", "True"]
    both = sys.argv[4].lower() in ["true", "True"]
    split = sys.argv[5]
    version = sys.argv[6]
    custom_name = sys.argv[7] if len(sys.argv) > 7 else None

    print(generate_save_name(config_file, bias, direct , both, split, version, custom_name))
    