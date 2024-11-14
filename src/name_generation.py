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

def generate_save_name(model, bias, both, split, version, custom_name=None):
    if custom_name:
        return custom_name

    if "microsoft" in model :
        prefix = "naive"
    elif "MoritzLaurer" in model:
        prefix = "mnli"
    else:
        prefix = f"{model.replace("/", "_")}_split{split}_v{version}"
    
    both_indicator = "2w" if both else "1w"
    bias_indicator = "biased" if bias else "unbiased"

    size = extract_size(model)

    save_name = f"{prefix}_derberta_{size}_{both_indicator}_{bias_indicator}_split{split}_v{version}"
    return save_name

if __name__ == "__main__":
    model = sys.argv[1]
    bias = sys.argv[2].lower() == "true"
    both = sys.argv[3]
    split = sys.argv[4]
    version = sys.argv[5]
    custom_name = sys.argv[6] if len(sys.argv) > 6 else None

    print(generate_save_name(model, bias, both, split, version, custom_name))
    