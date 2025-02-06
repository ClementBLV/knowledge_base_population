import argparse
import os
from pathlib import Path
import subprocess
import warnings

from src.base.compute_meta_probabilities import aggregate_probabilities, compute_meta_probabilities
from src.base.mnli_dataclass import MetaPredictionInputFeatures, PredictionInputFeatures, load_predictions
from src.base.compute_metrics import *
from src.utils.utils import str2bool, setup_logger


def compute_probabilities(args):
    """ Appelle compute_probabilities.py avec les arguments fournis. """
    command = [
        "python", "compute_probabilities.py",
        "--eval_file", args.eval_file,
        "--output_file", args.output_file,
        "--config_file", args.config_file,
        "--model", args.model,
        "--batch_size", str(args.batch_size),
        "--parallel", str(args.parallel),
        "--fast", str(args.fast)
    ]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()

    # Arguments pour le pipeline
    parser.add_argument("--direrct", type=str, help="Chemin vers les probabilités directes")
    parser.add_argument("--reverse", type=str, help="Chemin vers les probabilités inverses")
    parser.add_argument("--both_direct", type=str, help="Chemin vers les probabilités directes combinées")
    parser.add_argument("--both_reverse", type=str, help="Chemin vers les probabilités inverses combinées")
    parser.add_argument("--compute", action="store_true", help="Forcer le calcul des probabilités pour le modèle donné en entrée")
    parser.add_argument("--config_meta", type=str, help="Fichier de configuration méta")
    parser.add_argument("--meta_proba_file", type=str, help="Fichier de sortie pour les probabilités")
    parser.add_argument("--output_eval_name", type=str, help="Name of the eval file - the eval are saved in the folder eval")

    # Arguments pour compute_probabilities.py
    parser.add_argument("--eval_file", type=str, help="Fichier JSON d'entrée avec les relations")
    parser.add_argument("--proba_file", type=str, help="Fichier de sortie pour les probabilités")
    parser.add_argument("--config_file", type=str, help="Fichier de configuration du modèle")
    parser.add_argument("--model", type=str, help="Poids du modèle")
    parser.add_argument("--batch_size", type=int, default=32, help="Taille de batch")
    parser.add_argument("--parallel", type=str2bool, default=True, help="Exécution parallèle sur GPU")
    parser.add_argument("--fast", type=str2bool, default=False, help="Mode rapide")

    args = parser.parse_args()

    # logger 
    output_file_path = Path(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), args.output_eval_name))
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger( f"{output_file_path}.log")
    logger.info("Program: pipeline_eval.py ****")

    ### Simple config ###
    # We want to compute the probability for a given eval_file
    if args.compute :
        if not args.eval_file or not args.proba_file or not args.config_file or not args.model:
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
    if args.config_meta or (args.direrct and args.reverse and args.both_direct and args.both_reverse):
        if not all([args.direrct, args.reverse, args.both_direct, args.both_reverse]):
            parser.error("You should either give ONE probability file - FOUR probability file - ONE config_meta.json file")

        direct_probs = load_predictions(args.direrct, logger)
        reverse_probs = load_predictions(args.reverse, logger)
        both_direct_probs = load_predictions(args.both_direct, logger)
        both_reverse_probs = load_predictions(args.both_reverse, logger)

        # Appel à la fonction d’agrégation (à implémenter)
        aggregated_probs = aggregate_probabilities(direct_probs, reverse_probs, both_direct_probs, both_reverse_probs)
        print("Probabilités agrégées :", aggregated_probs[0])

        # Compute the probabilities and predictions
        compute_meta_probabilities(aggregated_probs, config_meta=args.config_meta ,meta_proba_file=args.meta_proba_file)
        predictions = load_predictions(args.proba_file,type_=MetaPredictionInputFeatures , logger=logger)

    # map the relation to their prediction 
    relations2predictions = {"Global":[]}
    for prediction in predictions : 
        relations2predictions["Global"].append(prediction.predictions)
        if prediction.relation not in relations2predictions.keys(): 
            relations2predictions[prediction.relation] = prediction.predictions
        else : 
            relations2predictions[prediction.relation].extend(prediction.predictions)
    
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
