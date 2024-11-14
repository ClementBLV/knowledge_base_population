import logging
import os
import random
import json
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass
from pprint import pprint
import shutil
import sys
from tqdm import tqdm
print("=========== SPLIT ============")

################ setup : logger ################
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logger.info("Progam : split.py ****")

parser = ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--bias", 
                    help="If true the output will be biased else it will not be")
parser.add_argument("--percentage", type=int, default=5,
                    help="The percentage 5 for 5% , 10 for 10%, correponding to the number of lines kept",)
parser.add_argument("--threshold_effectif", type=int, 
                    help="Minimal number of relation")
args = parser.parse_args()

assert ".json" in args.input_file, "File must be a json"
logger.info(f"File : {args.input_file}")
logger.info(f"Percetage kept : {args.percentage} %")

if args.bias == "false":
    bias = False
else:
    bias = True


@dataclass
class REInputFeatures:
    head_id: str
    tail_id: str
    premise: str
    subj: str
    obj: str
    context: str
    relation: str = None

if args.percentage not in [0, 100]: 
    if args.threshold_effectif is None or bias == True:
        logger.warning("The dataset will be biased")
        # random pickup
        with open(args.input_file, "rt") as f:
            mnli_data = []
            stats = []
            lines = json.load(f)
            logger.info(f"Number of inputs : {len(lines)}")
            logger.info(f"Number of ouputs : {round(len(lines) * args.percentage / 100)}")

            for line in tqdm(random.choices(lines, k=round(len(lines) * args.percentage / 100))):

                mnli_data.append(
                    REInputFeatures(
                        head_id=line["head_id"],
                        tail_id=line["tail_id"],
                        premise=line["premise"],
                        subj=line["subj"],
                        obj=line["obj"],
                        context=line["context"],
                        relation=line["relation"],
                    )
                )

            logger.info(f"Real percentage : {len(mnli_data) / len(lines)}")

    else:
        logger.warning("Homogeneous dataset")
        # homogeneous random picking throuout all the relations
        with open(args.input_file, "rt") as f:
            mnli_data = []
            stats = []
            lines = json.load(f)
            logger.info(f"Number of inputs : {len(lines)}")
            logger.info(f"Number of ouputs : {round(len(lines) * args.percentage / 100)}")
            relation2data = {}
            for line in lines:
                # accumulate all the elements for each relations
                if line["relation"] in relation2data:
                    relation2data[line["relation"]].append(
                        REInputFeatures(
                            head_id=line["head_id"],
                            tail_id=line["tail_id"],
                            premise=line["premise"],
                            subj=line["subj"],
                            obj=line["obj"],
                            context=line["context"],
                            relation=line["relation"],
                        )
                    )
                else:
                    relation2data[line["relation"]] = [
                        REInputFeatures(
                            head_id=line["head_id"],
                            tail_id=line["tail_id"],
                            premise=line["premise"],
                            subj=line["subj"],
                            obj=line["obj"],
                            context=line["context"],
                            relation=line["relation"],
                        )
                    ]
            # compute the effective of each data
            r2eff = {}
            for key, value in relation2data.items():
                num_elements = len(value)
                r2eff[key] = num_elements

            # pich up the relation according to the number of relation
            for rel, effective in r2eff.items():
                if effective == args.threshold_effectif:
                    mnli_data.extend(relation2data[rel])  # add all the relation
                if effective > args.threshold_effectif:
                    # if more data choose randomnly as much elt as the minimal bound
                    mnli_data.extend(
                        random.choices(relation2data[rel], k=args.threshold_effectif)
                    )
                # else the class is disgad
            # shuffle the final output
            if args.percentage == 0:
                # 0 means all so just suffle
                random.shuffle(mnli_data)

            else:
                mnli_data = random.choices(
                    mnli_data, k=round(len(lines) * args.percentage / 100)
                )
                    
    logger.info(f"Dataset size {len(mnli_data)}")


    # Ensure file exists
    output_file_path = Path(args.output_file)
    output_file_path.parent.mkdir(exist_ok=True, parents=True)

    # save
    json.dump(
        [data.__dict__ for data in mnli_data],
        open(args.output_file, "w", encoding="utf-8"),
        indent=4,
    )

else : 
    logger.info("Percentage choosen is 100% hence no Split")
    shutil.copy(args.input_file, args.output_file)

# Check if the file was created
if os.path.exists(args.output_file):
    logger.info(f"Save : tmp file {args.output_file} was successfully created.")
else:
    logger.error(f"Failed to create the file {args.output_file}.")