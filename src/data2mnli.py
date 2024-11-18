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
    FORBIDDEN_MIX,
    FB_LABEL_TEMPLATES
)
import os
print("=========== CONVERTION ============")

################ setup : seed ################
random.seed(0)
np.random.seed(0)
sys.path.append("./")

################ setup : logger ################
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.info("Progam : data2mnli.py ****")

################ setup : config ################
current_dir = os.path.dirname(__file__)
config_path = os.path.join(current_dir, "..", "config", "config.json")
with open(config_path, "r") as config_file:
    config = json.load(config_file)


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
parser.add_argument("--input_file", type=str)
parser.add_argument("--data_source", type=str, help="Folder with the data, used to pick the forbidden mixe relation")
parser.add_argument("--output_file", type=str)
parser.add_argument("--negn", type=int, default=1, help="Number of negative examples for each pair")
parser.add_argument("--direct", type=bool, default=True, help="If set on True only the direct relation will be present.")
parser.add_argument("--both", type=bool, default=False, help="If set on True the direct and inverse relation will be present")
parser.add_argument("--task", required=True, type=str, metavar="N", help="dataset name")
args = parser.parse_args()

################ setup : variables ################
logger.info(f"Task called: {args.task}")
logger.info(f"Convert {args.input_file} into NLI dataset")
assert (
        type(args.task) == type("str")
    ), "Must presise the dataset either 'wordnet', 'wn', 'wn18rr' or 'freebase', 'fb', 'fb15k237' in a string"
    
if args.task.lower() in ["wordnet", "wn", "wn18rr"]:
    LABELS = WN_LABELS
    LABEL_TEMPLATES = WN_LABEL_TEMPLATES
elif args.task.lower() in ["freebase", "fb", "fb15k237"]:
    LABELS = list(FB_LABEL_TEMPLATES.keys())
    LABEL_TEMPLATES = FB_LABEL_TEMPLATES
else : 
    raise TypeError("The task called is unknown")

# to correspond to the config of pretrained model - TO CHECK 
#labels2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
#{'entailment':0, 'not_entailment':1} # for deberta small which take the labels : ['entailment', 'not_entailment'] 
labels2id = config["labels2id"] 
logger.info(f"Label : the label used are {labels2id}")

################ setup : saving file ################
path = os.path.join(os.path.dirname(os.getcwd()), args.output_file)
output_file_path = Path(path)
output_file_path.parent.mkdir(exist_ok=True, parents=True)

################ setup : templates ################
templates = []
if args.direct and not (args.both):
    # direct case
    logger.info("Type : only direct relations")
    for relations in LABELS:
        templates.append(LABEL_TEMPLATES[relations][0])
        # the first ones are the direct labels
if not (args.direct) and not (args.both):
    # indirect case
    logger.info("Type : only reverse relations")
    for relations in LABELS:
        templates.append(LABEL_TEMPLATES[relations][1])
if args.both:
    logger.info("Type : both relations")
    for relations in LABELS:
        # this time we add both the direct and indirect
        templates.extend(LABEL_TEMPLATES[relations])


################ setup : positive-nagative examples ################

positive_templates: Dict[str, list] = defaultdict(list)
negative_templates: Dict[str, list] = defaultdict(list)
# generate a two dict,
# n°1 : positive_templates
#   {label of the relation in the dataset : "the positive template corresponding to this label"} (LABEL_TEMPLATES = positive_pattern)
# n°2 : negative_templates
#   {label of the relation in the dataset: "all the other pattern not related to this label, eg CONTRADCTION"}
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
                # not a relations with issues of similarity, it can be put as a contradiction
                negative_templates[relation].append(template)

            else:
                # relation wich need to can't be label as negative as hypernym and instance_hypernym
                if (
                    template not in FORBIDDEN_MIX[relation]
                ):  # avoidthe template to wich this relation is too close
                    negative_templates[relation].append(template)

# load the forbidden couples
if args.task in ["wordnet", "wn", "wn18rr"]:
    with open(
        os.path.join(
            f"{args.data_source}/preprocessed/forbidden_couples.json"
        ),
        "rt",
    ) as f:
        id2forbidden = json.load(f)  
        # { id_head : {id_tail_1 : [r1, r2], id_tail_2 : [r1, r2, r3, r4]} , id_head_2 : ...}}
else : 
    id2forbidden = {}


################ function : MNLI format ################
def data2mnli_with_negative_examples(
    instance: REInputFeatures,
    positive_templates,
    negative_templates,
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
    if str(instance.head_id) in id2forbidden.keys():
        # this entity as forbidden couples (else keep the random chossen no risk)
        if str(instance.tail_id) in id2forbidden[0][str(instance.head_id)].keys():
            # check if the tail of this head is present, which mean there are several relation between
            # this head and this tail and we need to check, else no pb the issues is with another tail
            n = 0
            # generate the forbidden template
            forbidden_templates = []
            for relation in id2forbidden[str(instance.head_id)][
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
    # TODO andle the contradiction case ? 
    mnli_instances.extend(
        [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id[
                    "not_entailment"
                ],  # remove the neutral part  # ["neutral"] if instance.label != "no_relation" else labels2id["contradiction"],
            )
            for t in negative_template
        ]
    )
    # make some examples where subj and obj are inverted
    return mnli_instances


data2mnli = data2mnli_with_negative_examples  # if args.negative_pattern else tacred2mnli

################ function : data population ################
with open(args.input_file, "rt") as f:
    mnli_data = []
    stats = []
    relations = []
    for line in json.load(f):
        mnli_instance = data2mnli(
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
            negn=args.negn,
        )
        mnli_data.extend(mnli_instance)
        relations.append(line["relation"])
        stats.append(line["relation"] != "no_relation")

################ function : save ################
## cf wn2eval pour corriger le bug
with open(args.output_file, "wt") as f:
    logger.info(f"Saveing : writing file : {args.output_file}")
    for data in mnli_data:
        f.write(f"{json.dumps(data.__dict__)}\n")
    # json.dump([data.__dict__ for data in mnli_data], f, indent=2)

json.dump(
    [data.__dict__ for data in mnli_data], open(path, "w", encoding="utf-8"), indent=4
)

logger.info(f"Saved at location : {args.output_file}")

count = Counter([data.label for data in mnli_data])
logger.info(f"Number of links : {count}")