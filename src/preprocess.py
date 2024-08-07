import os
import json
import argparse
import multiprocessing as mp

from multiprocessing import Pool
from typing import List

parser = argparse.ArgumentParser(description="preprocess")
parser.add_argument(
    "--task", default="wn18rr", type=str, metavar="N", help="dataset name"
)
parser.add_argument(
    "--workers", default=2, type=int, metavar="N", help="number of workers"
)
parser.add_argument(
    "--train-path", default="", type=str, metavar="N", help="path to training data"
)
parser.add_argument(
    "--valid-path", default="", type=str, metavar="N", help="path to valid data"
)
parser.add_argument(
    "--test-path", default="", type=str, metavar="N", help="path to valid data"
)

args = parser.parse_args()
mp.set_start_method("fork")


def _check_sanity(relation_id_to_str: dict):
    # We directly use normalized relation string as a key for training and evaluation,
    # make sure no two relations are normalized to the same surface form
    relation_str_to_id = {}
    for rel_id, rel_str in relation_id_to_str.items():
        if rel_str is None:
            continue
        if rel_str not in relation_str_to_id:
            relation_str_to_id[rel_str] = rel_id
        elif relation_str_to_id[rel_str] != rel_id:
            assert False, "ERROR: {} and {} are both normalized to {}".format(
                relation_str_to_id[rel_str], rel_id, rel_str
            )
    return


def _normalize_relations(examples: List[dict], normalize_fn, is_train: bool):
    relation_id_to_str = {}
    for ex in examples:
        if ex is not None: 
            rel_str = normalize_fn(ex["relation"])
            relation_id_to_str[ex["relation"]] = rel_str
            ex["relation"] = rel_str

    _check_sanity(relation_id_to_str)

    if is_train:
        out_path = '{}/relations.json'.format(os.path.dirname(args.train_path))
        with open(out_path, "w", encoding="utf-8") as writer:
            json.dump(relation_id_to_str, writer, ensure_ascii=False, indent=4)
            print("Save {} relations to {}".format(len(relation_id_to_str), out_path))


wn18rr_id2ent = {}


def _load_wn18rr_texts(path: str):
    global wn18rr_id2ent
    lines = open(path, "r", encoding="utf-8").readlines()
    for line in lines:
        fs = line.strip().split("\t")
        assert len(fs) == 3, "Invalid line: {}".format(line.strip())
        entity_id, word, desc = fs[0], fs[1].replace("__", ""), fs[2]
        wn18rr_id2ent[entity_id] = (entity_id, word, desc)
    print("Load {} entities from {}".format(len(wn18rr_id2ent), path))


def _process_line_wn18rr(line: str) -> dict:
    fs = line.strip().split("\t")
    assert len(fs) == 3, "Expect 3 fields for {}".format(line)
    head_id, relation, tail_id = fs[0], fs[1], fs[2]
    _, head, _ = wn18rr_id2ent[head_id]
    _, tail, _ = wn18rr_id2ent[tail_id]
    example = {
        "head_id": head_id,
        "head": head,
        "relation": relation,
        "tail_id": tail_id,
        "tail": tail,
    }
    return example


def preprocess_wn18rr(path, save=True):
    if not wn18rr_id2ent:
        _load_wn18rr_texts(
            "{}/wordnet-mlj12-definitions.txt".format(os.path.dirname(path))
        )
    lines = open(path, "r", encoding="utf-8").readlines()
    pool = Pool(processes=args.workers)
    examples = pool.map(_process_line_wn18rr, lines)
    pool.close()
    pool.join()

    _normalize_relations(
        examples,
        normalize_fn=lambda rel: rel,  # rel.replace('_', ' ').strip(),
        is_train=(path == args.train_path),
    )
    if save:
        out_path = path.replace(".txt", "_source.json")
        json.dump(
            examples,
            open(out_path, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=4,
        )
        print("Save {} examples to {}".format(len(examples), out_path))
    return examples


fb15k_id2ent = {}
fb15k_id2desc = {}

def _truncate(text: str, max_len: int):
    return ' '.join(text.split()[:max_len])

def _load_fb15k237_wikidata(path: str):
    global fb15k_id2ent, fb15k_id2desc
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 2, 'Invalid line: {}'.format(line.strip())
        entity_id, name = fs[0], fs[1]
        name = name.replace('_', ' ').strip()
        if entity_id in fb15k_id2desc:
            fb15k_id2ent[entity_id] = (entity_id, name, fb15k_id2desc.get(entity_id, ''))
        else: 
            # remove the print for log lisibility in train set out of 272115 pairs 684 disn't have match
            # those pairs were revoved in the dataset generation 
            print('No desc found for {}'.format(entity_id))
            continue
    print('Load {} entity names from {}'.format(len(fb15k_id2ent), path))


def _load_fb15k237_desc(path: str):
    global fb15k_id2desc
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 2, 'Invalid line: {}'.format(line.strip())
        entity_id, desc = fs[0], fs[1]
        fb15k_id2desc[entity_id] = _truncate(desc, 300)
    print('Load {} entity descriptions from {}'.format(len(fb15k_id2desc), path))


def _normalize_fb15k237_relation(relation: str) -> str:
    tokens = relation.replace('./', '/').strip().split('/')
    dedup_tokens = []
    for token in tokens:
        if token not in dedup_tokens[-3:]:
            dedup_tokens.append(token)
    # leaf words are more important (maybe)
    relation_tokens = dedup_tokens[::-1]
    relation = ' '.join([t for idx, t in enumerate(relation_tokens)
                         if idx == 0 or relation_tokens[idx] != relation_tokens[idx - 1]])
    return relation


def _process_line_fb15k237(line: str) -> dict:
    fs = line.strip().split('\t')
    assert len(fs) == 3, 'Expect 3 fields for {}'.format(line)
    head_id, relation, tail_id = fs[0], fs[1], fs[2]
    try : 
        _, head, _ = fb15k_id2ent[head_id]
        _, tail, _ = fb15k_id2ent[tail_id]
        example = {'head_id': head_id,
                'head': head,
                'relation': relation,
                'tail_id': tail_id,
                'tail': tail}
        return example
    except:
        print("No relation found for {} or {}".format(head_id, tail_id))
        return None


def preprocess_fb15k237(path):
    if not fb15k_id2desc:
        _load_fb15k237_desc('{}/FB15k_mid2description.txt'.format(os.path.dirname(path)))
    if not fb15k_id2ent:
        _load_fb15k237_wikidata('{}/FB15k_mid2name.txt'.format(os.path.dirname(path)))

    lines = open(path, 'r', encoding='utf-8').readlines()
    pool = Pool(processes=4) #args.workers)
    examples_ = pool.map(_process_line_fb15k237, lines)
    pool.close()
    pool.join()

    examples = [example for example in examples_ if example is not None]
    _normalize_relations(examples, normalize_fn=_normalize_fb15k237_relation, is_train=(path == path)) #args.train_path))

    out_path = path + '.json' # TODO remove .txt
    json.dump(examples, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print('Save {} examples to {}'.format(len(examples), out_path))

    # load the descripation
    json.dump(fb15k_id2desc, open(path + "_desc.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    return examples, fb15k_id2desc



def main():
    all_examples = []
    for path in [args.train_path, args.valid_path, args.test_path]:
        assert os.path.exists(path)
        print('Process {}...'.format(path))
        if args.task.lower() == 'wn18rr':
            all_examples += preprocess_wn18rr(path)
        elif args.task.lower() == 'fb15k237':
            all_examples += preprocess_fb15k237(path)
        else:
            assert False, "Unknown task: {}".format(args.task)

    if args.task.lower() == 'wn18rr':
        id2text = {k: v[2] for k, v in wn18rr_id2ent.items()}
    elif args.task.lower() == 'fb15k237':
        id2text = {k: v[2] for k, v in fb15k_id2ent.items()}
    else:
        assert False, 'Unknown task: {}'.format(args.task)

    dump_all_entities(
        all_examples,
        out_path='{}/entities.json'.format(os.path.dirname(args.train_path)),
        id2text=id2text)
    print('Done')



#if __name__ == "__main__":
#    main()
