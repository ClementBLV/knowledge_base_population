from argparse import ArgumentParser
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict
import numpy as np
import json
import sys
from pprint import pprint
import random
from templates import WN_LABELS, WN_LABEL_TEMPLATES , templates_direct, template_indirect, FORBIDDEN_MIX


random.seed(0)
np.random.seed(0)

sys.path.append("./")
#directly express in the code for WN
#from a2t.relation_classification.tacred import TACRED_LABEL_TEMPLATES, TACRED_LABELS


@dataclass
class REInputFeatures:
    subj: str
    obj: str
    context: str
    pair_type: str = None
    relation: str = None


@dataclass
class MNLIInputFeatures:
    premise: str #context
    hypothesis: str #relation 
    label: int


parser = ArgumentParser()

parser.add_argument("--input_file", type=str, default="data/WN18RR/valid.json")
parser.add_argument("--output_file", type=str, default="data/WN18RR/valid.mnli.json")
parser.add_argument("--negn", type=int, default=1)
parser.add_argument("--direct", type=bool, default=True)
parser.add_argument("--both", type=bool, default=True)

args = parser.parse_args()
print("=========== CONVERTION ============")
print("convert ", args.input_file , " into NLI dataset")


labels2id = {"entailment": 2, "neutral": 1, "contradiction": 0}

positive_templates: Dict[str, list] = defaultdict(list)
negative_templates: Dict[str, list] = defaultdict(list)

templates = []
if args.direct : 
    for relations in WN_LABELS: 
        templates.append(WN_LABEL_TEMPLATES[relations][0]) # the first ones are the direct labels
else : 
    for relations in WN_LABELS: 
        templates.append(WN_LABEL_TEMPLATES[relations][1])

# generate a two dict, 
# n°1 : positive_templates 
#   {label of the relation in the dataset : "the positive template corresponding to this label"} (LABEL_TEMPLATES = positive_pattern)
# n°2 : negative_templates
#   {label of the relation in the dataset: "all the other pattern not related to this label, eg contradiction"}
for relation in WN_LABELS: 
    #if not args.negative_pattern and label == "no_relation":
    #    continue
    for template in templates:
    # for the moment we just look at the direct patterns 
        if template in WN_LABEL_TEMPLATES[relation]:     
        # if the template is realy the one corresponding to the relation 
            positive_templates[relation].append(template)    # on lie le label de la relation aux template dans le dictionnaire des template { label : template }
        
        else:
            if relation not in FORBIDDEN_MIX.keys():
            # not a relations with issues of similarity 
                negative_templates[relation].append(template)

            else :
            # relation wich need to can't be label as negative as hypernym and instance_hypernym
                if template not in FORBIDDEN_MIX[relation]: # avoidthe template to wich this relation is too close 
                    negative_templates[relation].append(template)
            
def wn2mnli_with_negative_pattern(
    instance: REInputFeatures,
    positive_templates,
    negative_templates,
    templates,
    negn=1,
    posn=1,
):
    mnli_instances = []
    # Generate the positive examples

    positive_template = random.choices(positive_templates[instance.relation], k=posn)
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
    negative_template = random.choices(negative_templates[instance.relation], k=negn) 
    mnli_instances.extend(
        [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id["contradiction"], # remove the neutral part  # ["neutral"] if instance.label != "no_relation" else labels2id["contradiction"],
            )
            for t in negative_template
        ]
    )
    
    return mnli_instances


wn2mnli = wn2mnli_with_negative_pattern #if args.negative_pattern else tacred2mnli


with open(args.input_file, "rt") as f:
    mnli_data = []
    stats = []
    relations =[]
    for line in json.load(f):
        mnli_instance = wn2mnli(
            REInputFeatures(
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

## cf wn2eval pour corriger le bug
with open(args.output_file, "wt") as f:
    for data in mnli_data:
        f.write(f"{json.dumps(data.__dict__)}\n")
    #json.dump([data.__dict__ for data in mnli_data], f, indent=2)

count = Counter([data.label for data in mnli_data])
print("Number of links : ", count)
count = Counter(relations)
pprint(dict(count))
