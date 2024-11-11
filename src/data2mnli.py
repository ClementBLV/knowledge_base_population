from argparse import ArgumentParser
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging
from typing import Dict
import numpy as np
import json
import sys
from pprint import pprint
import random
from pathlib import Path
from templates import (
    WN_LABELS,
    WN_LABEL_TEMPLATES,
    templates_direct,
    template_indirect,  # TODO andle the indirect case
    FORBIDDEN_MIX,
    FB_LABEL_TEMPLATES
)
import os

################ setup : config ################
random.seed(0)
np.random.seed(0)
sys.path.append("./")
logger = logging.getLogger(__name__)
logger.setup_logging()
# directly express in the code for WN
# from a2t.relation_classification.tacred import TACRED_LABEL_TEMPLATES, TACRED_LABELS

################ setup : data objects ################
@dataclass
class REInputFeatures:
    head_id: str
    tail_id: str
    subj: str
    obj: str
    context: str
    pair_type: str = None
    relation: str = None


@dataclass
class MNLIInputFeatures:
    premise: str  # context
    hypothesis: str  # relation
    label: int

################ setup : parser ################
parser = ArgumentParser()
parser.add_argument("--input_file", type=str, default="data/WN18RR/valid.json")
parser.add_argument("--output_file", type=str, default="data/WN18RR/valid.mnli.json")
parser.add_argument("--negn", type=int, default=1, help="Number of negative examples for each pair")
parser.add_argument("--direct", type=bool, default=True, help="If set on True only the direct relation will be present.")
parser.add_argument("--both", type=bool, default=False, help="If set on True the direct and inverse relation will be present")
parser.add_argument("--task", required=True, type=str, metavar="N", help="dataset name")
args = parser.parse_args()

assert (
        type(args.task) == type("str")
    ), "Must presise the dataset either 'wordnet', 'wn', 'wn18rr' or 'freebase', 'fb', 'fb15k237' in a string"
    
if args.task.lower() in ["wordnet", "wn", "wn18rr"]:
    LABELS = WN_LABELS
    LABEL_TEMPLATES = WN_LABEL_TEMPLATES
if args.task.lower() in ["freebase", "fb", "fb15k237"]:
    LABELS = list(FB_LABEL_TEMPLATES.keys())
    LABEL_TEMPLATES = FB_LABEL_TEMPLATES
else : 
    raise TypeError("The task called is unknown")

print("=========== CONVERTION ============")
print("convert ", args.input_file, " into NLI dataset")


# to correspond to the config of pretrained model
labels2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

positive_templates: Dict[str, list] = defaultdict(list)
negative_templates: Dict[str, list] = defaultdict(list)

templates = []
if args.direct and not (args.both):
    # direct case
    for relations in LABELS:
        templates.append(
            LABEL_TEMPLATES[relations][0]
        )  # the first ones are the direct labels
if not (args.direct) and not (args.both):
    # indirect case
    for relations in LABELS:
        templates.append(LABEL_TEMPLATES[relations][1])
if args.both:
    for relations in LABELS:
        # this time we add both the direct and indirect
        templates.extend(LABEL_TEMPLATES[relations])
# generate a two dict,
# n°1 : positive_templates
#   {label of the relation in the dataset : "the positive template corresponding to this label"} (LABEL_TEMPLATES = positive_pattern)
# n°2 : negative_templates
#   {label of the relation in the dataset: "all the other pattern not related to this label, eg contradiction"}
for relation in LABELS:
    # if not args.negative_pattern and label == "no_relation":
    #    continue
    for template in templates:
        # for the moment we just look at the direct patterns
        if template in LABEL_TEMPLATES[relation]:
            # if the template is realy the one corresponding to the relation
            positive_templates[relation].append(
                template
            )  # on lie le label de la relation aux template dans le dictionnaire des template { label : template }
            # if not both only 1 relation is added
            # else the direct and indirect aire added
        else:
            if relation not in FORBIDDEN_MIX.keys():
                # not a relations with issues of similarity
                negative_templates[relation].append(template)

            else:
                # relation wich need to can't be label as negative as hypernym and instance_hypernym
                if (
                    template not in FORBIDDEN_MIX[relation]
                ):  # avoidthe template to wich this relation is too close
                    negative_templates[relation].append(template)
# pprint(positive_templates)
# pprint(negative_templates)
# load the forbidden couples
if args.task in ["wordnet", "wn", "wn18rr"]:
    with open(
        os.path.join(
            os.path.dirname(os.getcwd()), "data/WN18RR/source/forbidden_couple.json"
        ),
        "rt",
    ) as f:
        id2forbidden = json.load(
            f
        )  # { id_head : {id_tail_1 : [r1, r2], id_tail_2 : [r1, r2, r3, r4]} , id_head_2 : ...}}
else : 
    id2forbidden = [{}]

def wn2mnli_with_negative_pattern(
    instance: REInputFeatures,
    positive_templates,
    negative_templates,
    templates,
    negn,
    posn=1,
):
    if args.both:
        negn, posn = 2, 2  # cause to examples for direct and indirect
    mnli_instances = []
    # Generate the positive examples
    if posn < len(positive_templates[instance.relation]):
        positive_template = random.choices(
            positive_templates[instance.relation], k=posn
        )
    else:  # no need to randomly pick up examples as all of them must be picked up
        positive_template = positive_templates[instance.relation]
    # print(positive_template)

    # add the templates to the relation
    mnli_instances.extend(
        [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id["entailment"],
            )
            for t in positive_template
        ]
    )

    # Generate the negative templates
    # random pick up in the negative templates here 1 ,
    # hense for each positive element there is a random negative one

    # while it is a forbidden couple : choose an other one :
    # init the neg template
    negative_template = random.choices(negative_templates[instance.relation], k=negn)
    # print(instance.head_id  id2forbidden[0].keys())
    if str(instance.head_id) in id2forbidden[0].keys():
        # this entity as forbidden couples (else keep the random chossen no risk)
        if str(instance.tail_id) in id2forbidden[0][str(instance.head_id)].keys():
            # check if the tail of this head is present, which mean there are several relation between
            # this head and this tail and we need to check, else no pb the issues is with another tail
            n = 0
            # generate the forbidden template
            forbidden_templates = []
            for relation in id2forbidden[0][str(instance.head_id)][
                str(instance.tail_id)
            ]:
                # for each relations in the forbidden mix we add the template
                forbidden_templates.extend(WN_LABEL_TEMPLATES[relation])
                print(forbidden_templates)
            # check it
            while negative_template in forbidden_templates and n < 10:
                negative_template = random.choices(
                    negative_templates[instance.relation], k=negn
                )
                n += 1

    mnli_instances.extend(
        [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id[
                    "contradiction"
                ],  # remove the neutral part  # ["neutral"] if instance.label != "no_relation" else labels2id["contradiction"],
            )
            for t in negative_template
        ]
    )
    # make some examples where subj and obj are inverted
    return mnli_instances


wn2mnli = wn2mnli_with_negative_pattern  # if args.negative_pattern else tacred2mnli


with open(args.input_file, "rt") as f:
    mnli_data = []
    stats = []
    relations = []
    for line in json.load(f):
        mnli_instance = wn2mnli(
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
            templates_direct,
            negn=args.negn,
        )
        mnli_data.extend(mnli_instance)
        relations.append(line["relation"])
        stats.append(line["relation"] != "no_relation")

path = os.path.join(os.path.dirname(os.getcwd()), args.output_file)

# Ensure file exists
output_file_path = Path(path)
output_file_path.parent.mkdir(exist_ok=True, parents=True)

## cf wn2eval pour corriger le bug
with open(args.output_file, "wt") as f:
    print(f"writing file : {args.output_file}")
    for data in mnli_data:
        f.write(f"{json.dumps(data.__dict__)}\n")
    # json.dump([data.__dict__ for data in mnli_data], f, indent=2)

# save
json.dump(
    [data.__dict__ for data in mnli_data], open(path, "w", encoding="utf-8"), indent=4
)

count = Counter([data.label for data in mnli_data])
print("Number of links : ", count)
count = Counter(relations)
pprint(dict(count))
print("saved at location : ", args.output_file)