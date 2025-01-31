"""
args : 
    tous les arguments nécéssaires pour appeler compute_probabilities.py
    + les args pour le pipeline de probabilité 
    direrct un chemin vers les probabilitées 
    reverse 
    both_direct
    both_reverse
    compute 
    config_meta

soit on calcule les probabilitées : 
    si l'utilisateur donne un chemin - compute = True
    appeler compute_probabilities.py avec les arguments suivants complétées : 
        parser = argparse.ArgumentParser()
        parser.add_argument("--eval_file", type=str, required=True, help="Input JSON file")
        parser.add_argument("--output_file", type=str, required=True, help="File to save computed probabilities")
        parser.add_argument("--config_file", type=str, required=True, help="Config file for the model")
        parser.add_argument("--model", type=str, required=True, help="Path to the model weights")
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size for probability computation")
        parser.add_argument("--parallel",    type=str2bool,    default=True,    help="If true the evaluation will be done in batch and parallelise on GPU",)
        parser.add_argument("--fast", type=str2bool, default=False, help="Use only 1000 for debug and fast test")
        args = parser.parse_args()
    on va lire le fichier de probabilités 

soit on recalcule pas les probabilitées:  
    on donne un chemin et on precise pas compute 
    si le fichier précisé n'existe pas on lève un warning et on lance le calcul des probabilitées à cet endroit 
    sinon on lis le ficher 

si on donne les 4 chemins des fichiers de probabilitées ou qu'on précise une meta config 
    utiliser le méta model - lire les 4 fichiers de probabilitées - appeler la fonction d'aggrégation 





"""


import argparse
import os
import subprocess
import json
import warnings


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


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


def load_probabilities(file_path):
    """ Charge les probabilités à partir d'un fichier JSON. """
    with open(file_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()

    # Arguments pour le pipeline
    parser.add_argument("--direrct", type=str, help="Chemin vers les probabilités directes")
    parser.add_argument("--reverse", type=str, help="Chemin vers les probabilités inverses")
    parser.add_argument("--both_direct", type=str, help="Chemin vers les probabilités directes combinées")
    parser.add_argument("--both_reverse", type=str, help="Chemin vers les probabilités inverses combinées")
    parser.add_argument("--compute", action="store_true", help="Forcer le calcul des probabilités")
    parser.add_argument("--config_meta", type=str, help="Fichier de configuration méta")

    # Arguments pour compute_probabilities.py
    parser.add_argument("--eval_file", type=str, help="Fichier JSON d'entrée")
    parser.add_argument("--output_file", type=str, help="Fichier de sortie pour les probabilités")
    parser.add_argument("--config_file", type=str, help="Fichier de configuration du modèle")
    parser.add_argument("--model", type=str, help="Poids du modèle")
    parser.add_argument("--batch_size", type=int, default=32, help="Taille de batch")
    parser.add_argument("--parallel", type=str2bool, default=True, help="Exécution parallèle sur GPU")
    parser.add_argument("--fast", type=str2bool, default=False, help="Mode rapide")

    args = parser.parse_args()

    # Vérification du calcul ou de la lecture des probabilités
    if args.compute:
        if not args.eval_file or not args.output_file or not args.config_file or not args.model:
            parser.error("--compute nécessite --eval_file, --output_file, --config_file et --model")
        compute_probabilities(args)
        probabilities = load_probabilities(args.output_file)
    else:
        if args.output_file and os.path.exists(args.output_file):
            probabilities = load_probabilities(args.output_file)
        else:
            warnings.warn(f"Le fichier {args.output_file} n'existe pas, calcul des probabilités en cours...")
            compute_probabilities(args)
            probabilities = load_probabilities(args.output_file)

    # Vérification de la méta configuration
    if args.config_meta or (args.direrct and args.reverse and args.both_direct and args.both_reverse):
        if not all([args.direrct, args.reverse, args.both_direct, args.both_reverse]):
            parser.error("Les 4 fichiers de probabilités doivent être fournis ou une configuration méta doit être précisée.")
        # Charger les fichiers et appeler la fonction d'agrégation
        direct_probs = load_probabilities(args.direrct)
        reverse_probs = load_probabilities(args.reverse)
        both_direct_probs = load_probabilities(args.both_direct)
        both_reverse_probs = load_probabilities(args.both_reverse)

        # Appel à la fonction d’agrégation (à implémenter)
        aggregated_probs = aggregate_probabilities(direct_probs, reverse_probs, both_direct_probs, both_reverse_probs)
        print("Probabilités agrégées :", aggregated_probs)


def aggregate_probabilities(direct, reverse, both_direct, both_reverse):
    """ Fonction d'agrégation des probabilités (à implémenter selon les besoins). """
    return {
        "direct": sum(direct.values()) / len(direct),
        "reverse": sum(reverse.values()) / len(reverse),
        "both_direct": sum(both_direct.values()) / len(both_direct),
        "both_reverse": sum(both_reverse.values()) / len(both_reverse),
    }


if __name__ == "__main__":
    main()
