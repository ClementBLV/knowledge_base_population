from argparse import ArgumentParser
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict
import numpy as np
import json
import sys
from pprint import pprint
import random

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
    relation: int


parser = ArgumentParser()

parser.add_argument("--input_file", type=str, default="data/WN18RR/valid.json")
# parser.add_argument('--model', type=str, default='microsoft/deberta-v2-xlarge-mnli')
parser.add_argument("--output_file", type=str, default="data/WN18RR/valid.mnli.json")
#parser.add_argument("--negative_pattern", action="store_true", default=False)
parser.add_argument("--negn", type=int, default=1)

args = parser.parse_args()
print("=========== CONVERTION ============")
print("convert ", args.input_file , " into NLI dataset")


WN_LABEL_TEMPLATES = { # in the future we will add the indirect relations 
    "_hypernym":["{obj} specifies {subj}"],
    "_derivationally_related_form":[ "{obj} derived from {subj}"],
    "_instance_hypernym":[ "{obj} is a {subj}"],
    "_also_see":[ "{obj} is seen in {subj}"],
    "_member_meronym":[ "{obj} is the family of {subj}" ],
    "_synset_domain_topic_of":[  "{obj} is a topic of {subj}"],
    "_has_part":[  "{obj} contains {subj}"],
    "_member_of_domain_usage":[ "{obj} ?????????? {subj}"],
    "_member_of_domain_region":[ "{obj} is the domain region of {subj}"],
    "_verb_group":[  "{obj} is synonym to {subj}"],
    "_similar_to":[ "{obj} is similar to {subj}"],
}

WN_LABELS = [
    "_hypernym",
    "_derivationally_related_form",
    "_instance_hypernym",
    "_also_see",
    "_member_meronym",
    "_synset_domain_topic_of",
    "_has_part",
    "_member_of_domain_usage",
    "_member_of_domain_region",
    "_verb_group",
    "_similar_to",
]

templates_direct = [
    "{obj} specifies {subj}",
    "{obj} derived from {subj}",
    "{obj} is a {subj}",
    "{obj} is seen in {subj}",
    "{obj} is the family of {subj}",
    "{obj} is a topic of {subj}",
    "{obj} contains {subj}",
    "{obj} ?????????? {subj}",
    "{obj} is the domain region of {subj}",
    "{obj} is synonym to {subj}", 
    "{obj} is similar to {subj}" 
]

template_indirect = [
    "{subj} generalize {obj}",
    "{	X }",
    "{subj} such as {obj}"
    "{subj} has 		 {obj}",
    "{subj} is a member of {obj}"
    "{subj} is the context of {obj}",
    "{subj} is a part of {obj}",
    "{obj} ?????????? 	    {subj}"
    "{subj} belong to the regieon of {obj}",
    "{subj} is synonym to {obj}",
    "{subj} similar to {obj}",
]

labels2id = {"entailment": 2, "neutral": 1, "contradiction": 0}

positive_templates: Dict[str, list] = defaultdict(list)
negative_templates: Dict[str, list] = defaultdict(list)


# generate a two dict, 
# n°1 : positive_templates 
#   {label of the relation in the dataset : "the positive template corresponding to this label"} (LABEL_TEMPLATES = positive_pattern)
# n°2 : negative_templates
#   {label of the relation in the dataset: "all the other pattern not related to this label, eg contradiction"}
for relation in WN_LABELS: #TACRED_LABELS:
    #if not args.negative_pattern and label == "no_relation":
    #    continue
    for template in templates_direct:
    # for the moment we just look at the direct patterns 
        if template in WN_LABEL_TEMPLATES[relation  ]:     # si le template correspond au label du template  
            positive_templates[relation].append(template)    # on lie le label de la relation aux template dans le dictionnaire des template { label : template }
        
        else:
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
                relation=labels2id["entailment"],
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
                relation=labels2id["contradiction"], # remove the neutral part  # ["neutral"] if instance.label != "no_relation" else labels2id["contradiction"],
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

with open(args.output_file, "wt") as f:
    for data in mnli_data:
        f.write(f"{json.dumps(data.__dict__)}\n")
    json.dump([data.__dict__ for data in mnli_data], f, indent=2)

count = Counter([data.relation for data in mnli_data])
print("Number of links : ", count)
count = Counter(relations)
pprint(dict(count))
