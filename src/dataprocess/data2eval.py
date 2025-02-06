from argparse import ArgumentParser
from dataclasses import dataclass
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict
import numpy as np
import json
import sys
from src.utils.utils import setup_logger_basic
import utils.templates as templates
from base.mnli_dataclass import EvalInputFeatures
import os


from utils.templates import (
    WN_LABELS,
    WN_LABEL_TEMPLATES,
    FORBIDDEN_MIX,
    FB_LABEL_TEMPLATES
)

print("=========== CONVERTION ============")

################ setup : logger ################
logger = setup_logger_basic()
logger.info("Progam : data2eval.py ****")

sys.path.append("./")
# directly express in the code for WN
# from a2t.relation_classification.tacred import TACRED_LABEL_TEMPLATES, TACRED_LABELS


@dataclass
class REInputFeatures:
    subj: str
    obj: str
    context: str
    pair_type: str = None
    relation: str = None



parser = ArgumentParser()

parser.add_argument("--input_file", type=str, default="data/WN18RR/source/valid.json")
parser.add_argument("--output_file", type=str, default="data/WN18RR/valid_eval.json")
parser.add_argument("--direct", type=bool, default=True)
parser.add_argument("--task", type=str, default="fb")

args = parser.parse_args()
logger.info(f"convert {args.input_file} into NLI dataset")


# Initialize the lists for direct and indirect templates
direct_templates = []
indirect_templates = []

# choose the right label 
if args.task.lower() in ["wordnet", "wn", "wn18rr"]:
    LABELS = WN_LABELS
    LABEL_TEMPLATES = WN_LABEL_TEMPLATES
    direct_templates = templates.templates_direct
    indirect_templates = templates.template_indirect
if args.task.lower() in ["freebase", "fb", "fb15k237"]:
    LABELS = list(FB_LABEL_TEMPLATES.keys())
    LABEL_TEMPLATES = FB_LABEL_TEMPLATES
    # Loop through the dictionary and separate the templates

    for key, templates in FB_LABEL_TEMPLATES.items():
        if len(templates) >= 2:
            direct_templates.append(templates[0])
            indirect_templates.append(templates[1])


# labels2id = {"entailment": 2, "neutral": 1, "contradiction": 0}
labels2id = {'entailment':0, 'not_entailment':1} # for deberta small which take the labels : ['entailment', 'not_entailment'] 
logger.info(f"Label : the label used are {labels2id}")

positive_templates: Dict[str, list] = defaultdict(list)
negative_templates: Dict[str, list] = defaultdict(list)

if args.direct:
    templates = direct_templates
else:
    templates = indirect_templates

# generate a two dict,
# n°1 : positive_templates
#   {label of the relation in the dataset : "the positive template corresponding to this label"} (LABEL_TEMPLATES = positive_pattern)
# n°2 : negative_templates
#   {label of the relation in the dataset: "all the other pattern not related to this label, eg contradiction"}
for relation in LABELS:  # TACRED_LABELS:
    # if not args.negative_pattern and label == "no_relation":
    #    continue
    for template in templates:
        # for the moment we just look at the direct patterns
        if (template in LABEL_TEMPLATES[relation]):  
            # si le template correspond au label du template
            positive_templates[relation].append(template)  
            # on lie le label de la relation aux template dans le dictionnaire des template { label : template }

        else:
            negative_templates[relation].append(template)


def wn2mnli_eval(
    instance: REInputFeatures,
    positive_templates: Dict[str, list],
    negative_templates: Dict[str, list],
):
    mnli_instances_true = []
    mnli_instances_false = []

    # Generate the positive case
    mnli_instances_true.extend(
        [
            f"{t.format(subj=instance.subj, obj=instance.obj)}."
            for t in positive_templates[instance.relation]
        ]
    )

    # Generate ALL the negative templates
    mnli_instances_false.extend(
        [
            f"{t.format(subj=instance.subj, obj=instance.obj)}."
            for t in negative_templates[instance.relation]  
        ] # to have all the negative ones always in the same order for a given true relation
    )

    return EvalInputFeatures(
        premise=instance.context,
        hypothesis_true=mnli_instances_true[0],
        hypothesis_false=mnli_instances_false,
        relation=instance.relation,
    )


# Check if the directory exists, create if not
path = os.path.join(os.path.dirname(os.getcwd()), args.input_file)
with open(path, "rt") as f:
    mnli_data = []

    for line in json.load(f):
        mnli_feature = wn2mnli_eval(
            REInputFeatures(
                subj=line["subj"],
                obj=line["obj"],
                context=line["context"],
                relation=line["relation"],
            ),
            positive_templates,
            negative_templates,
        )

        mnli_data.append(mnli_feature)

path = os.path.join(os.path.dirname(os.getcwd()), args.output_file)

# Ensure file exists
output_file_path = Path(path)
output_file_path.parent.mkdir(exist_ok=True, parents=True)

with open(path, "wt") as f:
    logger.info(f"Saving : Writing file : {args.output_file}")
    for data in mnli_data:
        f.write(f"{json.dumps(data.__dict__)}\n")
    # json.dump([data.__dict__ for data in mnli_data], f, indent=2)

# save
json.dump(
    [data.__dict__ for data in mnli_data], open(path, "w", encoding="utf-8"), indent=4
)
logger.info(f"Save : saved at location : {args.output_file}")
