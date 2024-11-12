import sys
import warnings
import logger 
import logging 

logger = logging.getLogger(__name__)
logger.setup_logging()

def generate_save_name(model, bias, both, split, version, custom_name=None):
    if custom_name:
        return custom_name

    if model == "microsoft/deberta-v3-base":
        prefix = "untrained"
    elif model == "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli":
        prefix = "trained"
    else:
        logger.warning(f"Unknown model: {model}. Defaulting to 'unknown_model'.")
        prefix = "unknown_model"
    
    both_indicator = "2w" if both else "1w"
    bias_indicator = "biased" if bias else "unbiased"

    save_name = f"{prefix}_{both_indicator}_derbertabase_{bias_indicator}_split{split}_v{version}"
    logger.info(f"Saving Name : {save_name}")
    return save_name

if __name__ == "__main__":
    model = sys.argv[1]
    bias = sys.argv[2].lower() == "true"
    both = sys.argv[3]
    split = sys.argv[4]
    version = sys.argv[5]
    custom_name = sys.argv[6] if len(sys.argv) > 6 else None

    print(generate_save_name(model, bias, both, split, version, custom_name))
