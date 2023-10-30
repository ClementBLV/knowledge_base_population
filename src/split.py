import random
import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pprint import pprint 
parser = ArgumentParser()

parser.add_argument("--input_file", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--percentage", type=int, default=5, 
                    help= "The percentage 5 for 5% , 10 for 10%, correponding to the number of lines kept")

args = parser.parse_args()

assert (".json" in args.input_file), "File must be a json"
print("=========== SPLIT ============")
print("File : ", args.input_file)
print("Percetage kept : ", args.percentage , "%")


@dataclass
class REInputFeatures:
    premise:str
    subj: str
    obj: str
    context: str
    relation: str = None


# random pickup 
with open(args.input_file, "rt") as f:
    mnli_data = []
    stats = []
    lines = json.load(f)
    print("Number of inputs : " , len(lines))
    print("Number of ouputs : " , round(len(lines)*args.percentage/100))
                          
    for line in random.choices(lines, k=round(len(lines)*args.percentage/100)):

        mnli_data.append(
            REInputFeatures(
                premise=  line["premise"],
                subj =  line["subj"],
                obj =  line["obj"],
                context   =  line["context"],
                relation =  line["relation"],
                )
        )
        
    print("Real percentage : " , len(mnli_data)/len(lines))

# save
json.dump([data.__dict__ for data in mnli_data], open(args.output_file, "w",encoding='utf-8'), indent=4)
print("saved at location : ", args.output_file)

