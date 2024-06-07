from argparse import ArgumentParser
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, List
import numpy as np
import json
import sys
from pprint import pprint
import random
from templates import (
    WN_LABELS,
    WN_LABEL_TEMPLATES,
    templates_direct,
    template_indirect,
    FORBIDDEN_MIX,
    FB_LABEL_TEMPLATES
)
random.seed(0)
np.random.seed(0)

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


@dataclass
class MNLIInputFeatures:
    premise: str  # context
    hypothesis_true: List[str]
    hypothesis_false: List[str]
    relation: str


parser = ArgumentParser()

parser.add_argument("--input_file", type=str, default="data/WN18RR/source/valid.json")
parser.add_argument("--output_file", type=str, default="data/WN18RR/valid_eval.json")
parser.add_argument("--direct", type=bool, default=True)
parser.add_argument("--task", type=str, default="fb")

args = parser.parse_args()
print("=========== CONVERTION ============")
print("convert ", args.input_file, " into NLI dataset")

# choose the right label 
if args.task.lower() in ["wordnet", "wn", "wn18rr"]:
    LABELS = WN_LABELS
    LABEL_TEMPLATES = WN_LABEL_TEMPLATES
if args.task.lower() in ["freebase", "fb", "fb15k237"]:
    LABELS = list(FB_LABEL_TEMPLATES.keys())
    LABEL_TEMPLATES = FB_LABEL_TEMPLATES

# labels2id = {"entailment": 2, "neutral": 1, "contradiction": 0}
labels2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

positive_templates: Dict[str, list] = defaultdict(list)
negative_templates: Dict[str, list] = defaultdict(list)

if args.direct:
    templates = templates_direct
else:
    templates = template_indirect

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
        if (
            template in LABEL_TEMPLATES[relation]
        ):  # si le template correspond au label du template
            positive_templates[relation].append(
                template
            )  # on lie le label de la relation aux template dans le dictionnaire des template { label : template }

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
            for t in negative_templates[
                instance.relation
            ]  # to have all the negative ones
        ]
    )
    return MNLIInputFeatures(
        premise=instance.context,
        hypothesis_true=mnli_instances_true,
        hypothesis_false=mnli_instances_false,
        relation=instance.relation,
    )


import os

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
with open(path, "wt") as f:
    for data in mnli_data:
        f.write(f"{json.dumps(data.__dict__)}\n")
    # json.dump([data.__dict__ for data in mnli_data], f, indent=2)

# save
json.dump(
    [data.__dict__ for data in mnli_data], open(path, "w", encoding="utf-8"), indent=4
)
print("saved at location : ", args.output_file)
