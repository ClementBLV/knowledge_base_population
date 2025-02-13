import argparse
import os
from pathlib import Path
import subprocess
import warnings

from src.base.compute_meta_probabilities import aggregate_probabilities, compute_meta_probabilities
from src.base.mnli_dataclass import MetaPredictionInputFeatures, PredictionInputFeatures, load_predictions
from src.base.compute_metrics import *
from src.utils.utils import get_config, str2bool, setup_logger


def compute_probabilities(args):
    """ Appelle compute_probabilities.py avec les arguments fournis. """
    script_path = os.path.join("src", "base", "compute_probabilities.py")  # Correct path

    command = [
        "python", script_path  ,
        "--eval_file", args.eval_file,
        "--proba_file", args.proba_file,
        "--config_file", args.config_file,
        "--model_weight_path", args.model_weight_path,
        "--batch_size", str(args.batch_size),
        "--parallel", str(args.parallel),
        "--fast", str(args.fast)
    ]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()

    # Arguments pour le pipeline
    # TODO add them in the config_meta
    parser.add_argument("--direct", type=str, help="Path to direct probabilities")
    parser.add_argument("--reverse", type=str, help="Path to reverse probabilities")
    parser.add_argument("--both_direct", type=str, help="Path to combined direct probabilities")
    parser.add_argument("--both_reverse", type=str, help="Path to combined reverse probabilities")
    parser.add_argument("--compute", action="store_true", help="Force probability computation for the given input model")
    parser.add_argument("--config_meta", type=str, help="Meta configuration file")
    parser.add_argument("--meta_proba_file", type=str, help="Output file for probabilities")
    parser.add_argument("--output_eval_name", type=str, help="Name of the eval file - the evals are saved in the eval folder")

    # Arguments for compute_probabilities.py
    # 
    parser.add_argument("--eval_file", type=str, help="Input JSON file with relations")
    parser.add_argument("--proba_file", type=str, help="Output file for probabilities")
    parser.add_argument("--config_file", type=str, help="Model configuration file")
    parser.add_argument("--model_weight_path", type=str, help="Model weight path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--parallel", type=str2bool, default=True, help="Parallel execution on GPU")
    parser.add_argument("--fast", type=str2bool, default=False, help="Fast mode")

    # TODO push the eval file in git for direct and indirect

    args = parser.parse_args()

    # logger 
    output_file_path = Path(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), args.output_eval_name))
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger( f"{output_file_path}.log")
    logger.info("Program: pipeline_eval.py ****")

    ### Simple config ###
    # We want to compute the probability for a given eval_file
    if args.compute :
        if not args.eval_file or not args.proba_file or not args.config_file or not args.model_weight_path:
            parser.error("--compute nécessite --eval_file, --proba_file, --config_file et --model")
        
        compute_probabilities(args)
        predictions = load_predictions(args.proba_file, type_=PredictionInputFeatures, logger=logger )
    
    else:
        if args.proba_file and os.path.exists(args.proba_file):
            predictions = load_predictions(args.proba_file, type_=PredictionInputFeatures, logger=logger )
        else:
            warnings.warn(f"Le fichier {args.proba_file} n'existe pas, calcul des probabilités en cours...")
            compute_probabilities(args)
            predictions = load_predictions(args.proba_file, type_=PredictionInputFeatures, logger=logger )

    ### Meta config ###
    if args.config_meta or (args.direct and args.reverse and args.both_direct and args.both_reverse):
        if not all([args.direct, args.reverse, args.both_direct, args.both_reverse]):
            parser.error("You should either give ONE probability file - FOUR probability file - ONE config_meta.json file")

        direct_probs = load_predictions(args.direct,type_=PredictionInputFeatures, logger=logger)
        reverse_probs = load_predictions(args.reverse,type_=PredictionInputFeatures, logger=logger)
        both_direct_probs = load_predictions(args.both_direct, type_=PredictionInputFeatures,logger=logger)
        both_reverse_probs = load_predictions(args.both_reverse,type_=PredictionInputFeatures, logger=logger)

        # Appel à la fonction d’agrégation (à implémenter)
        aggregated_probs = aggregate_probabilities(direct_probs, reverse_probs, both_direct_probs, both_reverse_probs)
        print("Probabilités agrégées :", aggregated_probs[0])

        # Compute the probabilities and predictions
        config_meta = get_config(args.config_meta)
        compute_meta_probabilities(aggregated_probs, config_meta=config_meta,meta_proba_file=args.meta_proba_file, logger=logger)
        predictions = load_predictions(args.proba_file,type_=MetaPredictionInputFeatures , logger=logger)

    # map the relation to their prediction 
    relations2predictions = {"Global":[]}
    for prediction in predictions : 
        relations2predictions["Global"].append(prediction.predictions)
        if prediction.relation not in relations2predictions.keys(): 
            relations2predictions[prediction.relation] = [prediction.predictions]
        else : 
            relations2predictions[prediction.relation].append(prediction.predictions)
    
    # Display the results for each relation
    for relation in relations2predictions.keys():
        # Compute the MRR
        mrr_relation = compute_mrr(relations2predictions[relation])
        logger.info(f"MRR;{relation};{mrr_relation}")

        # Compute Hit at 1-3-10 for the element across shots
        hits = compute_hits_at_k(relations2predictions[relation])
        for k, hit in hits.items():
            logger.info(f"Hit_at_{k};{relation};{hit/len(relations2predictions[relation])}")



if __name__ == "__main__":
    main()
