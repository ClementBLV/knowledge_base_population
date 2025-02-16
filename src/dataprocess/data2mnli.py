import argparse
import logging
import os
import random
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, List
from src.utils.templates import (
    WN_LABELS,
    WN_LABEL_TEMPLATES,
    FORBIDDEN_MIX,
    FB_LABEL_TEMPLATES,
)
from src.utils.utils import setup_logger_basic, str2bool

LABELS, LABEL_TEMPLATES = None , None


# Set random seeds
random.seed(0)

################ setup : dataclass ################
@dataclass
class REInputFeatures:
    """Represents a relation extraction input instance."""
    head_id: str
    tail_id: str
    subj: str
    obj: str
    context: str
    pair_type: str = None
    relation: str = None


@dataclass
class MNLIInputFeatures:
    """Represents an MNLI input instance."""
    premise: str
    hypothesis: str
    label: int
    way: int


@dataclass
class Relation:
    """Represents a relation and its template."""
    relation: str
    template: str
    way: int


################ setup : args ################
  
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert data into MNLI format.")
    parser.add_argument("--input_file", type=str, required=True, help="Input file path.")
    parser.add_argument("--data_source", type=str, required=True, help="Folder with data for forbidden relations.")
    parser.add_argument("--config_name", type=str, required=True, help="Name of the config file (JSON).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output file.")
    parser.add_argument("--negn", type=int, default=1, help="Number of negative examples per pair.")
    parser.add_argument("--direct", type=str2bool, default="true",  help="Include only direct relations.")
    parser.add_argument("--both", type=str2bool, default="false" , help="Include both direct and inverse relations.")
    parser.add_argument("--task", type=str, required=True, help="Dataset name (e.g., 'wordnet' or 'freebase').")
    return parser.parse_args()


################ setup : logger ################
logger = setup_logger_basic()

################ setup : config ################
def load_config(config_file: Path) -> Dict:
    with open(config_file, "r") as f:
        config = json.load(f)
    assert "label2id" in config, "Config must include 'label2id'."
    return config

################ setup : templates ################
# Create templates based on task and arguments
def create_templates(task: str, direct: bool, both: bool) -> List[Relation]:
    global LABELS
    global LABEL_TEMPLATES
    if task.lower() in ["wordnet", "wn", "wn18rr"]:
        LABELS = WN_LABELS
        LABEL_TEMPLATES = WN_LABEL_TEMPLATES
    elif task.lower() in ["freebase", "fb", "fb15k237"]:
        LABELS = list(FB_LABEL_TEMPLATES.keys())
        LABEL_TEMPLATES = FB_LABEL_TEMPLATES
    else:
        raise ValueError(f"Unknown task: {task}")

    templates = []
    if direct and not both:
        logger.info("Data: Only DIRECT relations")
        for relation in LABELS:
            templates.append(Relation(relation, LABEL_TEMPLATES[relation][0], way=1))
    elif not direct and not both:
        logger.info("Data: Only INDIRECT relations")
        for relation in LABELS:
            templates.append(Relation(relation, LABEL_TEMPLATES[relation][1], way=-1))
    elif both:
        logger.info("Data: DIRECT and INDIRECT relations")
        for relation in LABELS:
            templates.append(Relation(relation, LABEL_TEMPLATES[relation][0], way=1))
            templates.append(Relation(relation, LABEL_TEMPLATES[relation][1], way=-1))

    return templates

################ function : MNLI format ################
def format_relation(t: Relation, obj: str , subj:str , way: int)-> str:
    return f"{t.template.format(subj=subj, obj=obj)}."

# Generate MNLI examples with positive and negative templates
def data2mnli_with_negative_examples(
    instance: REInputFeatures,
    positive_templates: Dict[str, List[Relation]],
    negative_templates: Dict[str, List[Relation]],
    id2forbidden: Dict[str, Dict[str, List[str]]],
    label2id: Dict[str, int],
    way: int, 
    both: bool,
    negn: int=1,
) -> List[MNLIInputFeatures]:
    mnli_instances = []

    # Positive examples
    relevant_positive_templates = [
        t for t in positive_templates[instance.relation]
        if t.way == way or both
    ]
    mnli_instances.extend([
        MNLIInputFeatures(
            premise=instance.context,
            hypothesis=format_relation(t, instance.obj, instance.subj, t.way) ,
            label=label2id["entailment"],
            way=t.way
        )
        for t in relevant_positive_templates
    ])

    # Negative examples
    forbidden_templates = set(
        template
        for r in id2forbidden.get(instance.head_id, {}).get(instance.tail_id, [])
        for template in LABEL_TEMPLATES[r]
    ) # andle the case LABEL_TEMPLATE is empty
    # id2forbidden { id_head : {id_tail_1 : [r1, r2], id_tail_2 : [r1, r2, r3, r4]} , id_head_2 : ...}}

    potential_negatives = [
        t for t in negative_templates[instance.relation]
        if (t.template not in forbidden_templates and (t.way == way or both))
    ]
    selected_negatives = random.sample(
        potential_negatives, k=min(negn, len(potential_negatives))
    )
    
    # If both is enabled, add reverse counterparts for the selected negatives
    if both:
        reverse_negatives = [
            Relation(
                relation=t.relation,
                template=LABEL_TEMPLATES[t.relation][0 if -t.way > 0 else 1],
                way=-t.way
            )
            for t in selected_negatives
        ]
        selected_negatives.extend(reverse_negatives)

    mnli_instances.extend([
        MNLIInputFeatures(
            premise=instance.context,
            hypothesis=format_relation(t, instance.obj, instance.subj, t.way),
            label=label2id["not_entailment"],
            way=t.way
        )
        for t in selected_negatives
    ])
    return mnli_instances




def main():
    print("=========== CONVERTION ============")
    logger.info("Program : data2mnli.py ****")
    args = parse_args()

    ################ setup : param ################
    current_dir = Path(__file__).resolve().parent.parent
    config_file = current_dir.parent / "configs" / args.config_name
    config = load_config(config_file)
    label2id = config["label2id"]
    way = 1 if args.direct else -1 

    ################ setup : templates ################
    templates = create_templates(args.task, args.direct, args.both)
    assert LABELS is not None and LABEL_TEMPLATES is not None, 'Issue with template initialization'
    # Split templates into positive and negative examples
    # generate a two dict,
    # n°1 : positive_templates
    #   {label of the relation in the dataset : "the positive template corresponding to this label"} 
    #   (LABEL_TEMPLATES = positive_pattern(x1))} 
    #   eg : 
    #       {   "_hypernym":                    "{obj} specifies {subj}",
    #           "_derivationally_related_form": "{obj} derived from {subj}", ...
    # 
    # n°2 : negative_templates
    #   {label of the relation in the dataset: "all the other pattern not related to this label, 
    #    eg : 
    #       {   "_hypernym":                    ["{obj} derived from {subj}", "{obj} is the domain region of {subj}", ...], 
    #           "_derivationally_related_form": ["{obj} specifies {subj}",, ... ]

    positive_templates: Dict[str, List[Relation]] = defaultdict(list)
    negative_templates: Dict[str, List[Relation]] = defaultdict(list)

    for relation in LABELS:
        for template in templates:
            if template.template in LABEL_TEMPLATES[relation]:
                # Add the template as positive for the relation
                positive_templates[relation].append(template)
            else:
                if relation not in FORBIDDEN_MIX.keys():
                    # Add the template as negative if no forbidden mix is defined
                    negative_templates[relation].append(template)
                else:
                    # Skip templates that are too similar to the relation
                    if template.template not in FORBIDDEN_MIX[relation]:
                        negative_templates[relation].append(template)

 
    ################ setup : forbidden relation ################
    if args.task.lower() in ["wordnet", "wn", "wn18rr"]:
        forbidden_file = Path(args.data_source) / "preprocessed" / "forbidden_couples.json"
        with open(forbidden_file, "r") as f:
            id2forbidden = json.load(f)
    else:
        id2forbidden = {}

    ################ setup : output ################
    output_file_path = Path(os.path.join(os.path.dirname(os.getcwd()), args.output_file))
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Data population
    mnli_data = []
    relations = []
    stats = []

    with open(args.input_file, "rt") as f:
        input_data = json.load(f)
        logger.info(f"Loaded {len(input_data)} instances from {args.input_file}.")

        for line in input_data:
            mnli_instance = data2mnli_with_negative_examples(
                REInputFeatures(
                    head_id=line["head_id"],
                    tail_id=line["tail_id"],
                    subj=line["subj"],
                    obj=line["obj"],
                    context=line["context"],
                    relation=line["relation"],
                ),
                positive_templates,
                negative_templates,
                id2forbidden,
                label2id,
                way=way,
                negn=args.negn,
                both=args.both,
            )
            mnli_data.extend(mnli_instance)
            relations.append(line["relation"])
            stats.append(line["relation"] != "no_relation")

    # Save results
    logger.info(f"Saving {len(mnli_data)} MNLI instances to {args.output_file}.")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump([data.__dict__ for data in mnli_data], f, indent=4)

    count = Counter([data.label for data in mnli_data])
    logger.info(f"Label distribution: {count}")
if __name__ == "__main__":
    main()
