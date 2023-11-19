import preprocess
import pandas as pd
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import argparse
import json
import pathlib


parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument('--task', default='wn18rr', type=str, metavar='N',
                    help='dataset name')
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of workers')
parser.add_argument('--train-path', default='./data/WN18RR/train.txt', type=str, metavar='N',
                    help='path to training data')
parser.add_argument('--valid-path', default='./data/WN18RR/valid.txt', type=str, metavar='N',
                    help='path to valid data')
parser.add_argument('--test-path', default='./data/WN18RR/test.txt', type=str, metavar='N',
                    help='path to test data')

args = parser.parse_args()

def generate_nli_data(path): 
    head2tail_dict = preprocess.preprocess_wn18rr(path)
    head2tail_df = pd.DataFrame(head2tail_dict) 
    hypos2premises = []
    for index in range(0, len(head2tail_df)) :
        triplet = head2tail_df.iloc[index]
        hypo2premise = {}  
        s_head = get_synset(triplet["head"],triplet["head_id"])
        s_tail = get_synset(triplet["tail"], triplet["tail_id"])

        l_head = " or ".join([ lemma.name().replace('_' , ' ').strip() for lemma in s_head.lemmas()][0:2])
        l_tail = " or ".join([ lemma.name().replace('_' , ' ').strip() for lemma in s_tail.lemmas()][0:2])

        hypo2premise["head_id"] =triplet["head_id"]
        hypo2premise["tail_id"] =triplet["tail_id"]
        hypo2premise["context"] = l_head + " means " + s_head.definition() + ". " + l_tail + " means " + s_tail.definition()
        hypo2premise["premise"] = l_head + " <" + str( triplet["relation"]) + "> " + l_tail
        hypo2premise["subj"]=l_tail
        hypo2premise["obj"]=l_head
        hypo2premise["relation"]=triplet["relation"]
        hypos2premises.append(hypo2premise)
    
    out_path = path.replace('txt','json')
    json.dump(hypos2premises, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print('Save {} hypothesis and premises to {}'.format(len(hypos2premises), out_path))
    return hypos2premises

            
def get_synset(name, id):
        if "NN" in name : 
            s = wn.synset_from_pos_and_offset('n', int(id))
            return s

        if "VB" in name : 
            s = wn.synset_from_pos_and_offset('v', int(id))
            return s

        if "JJ" in name : 
            s = wn.synset_from_pos_and_offset('a', int(id))
            return s
        if "RB" in name : 
            s = wn.synset_from_pos_and_offset('r', int(id))
            return s

        else : 
            print ("+"+name)


def main():
    """path = "./data/WN18RR/test.txt"
    hypos2premises = generate_nli_data(path)
    print(pd.DataFrame(hypos2premises))"""

    generate_nli_data(args.train_path)
    generate_nli_data(args.test_path)
    generate_nli_data(args.valid_path)

    
if __name__ == '__main__':
    main()

        