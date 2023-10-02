#!/usr/bin/env python3
import wn
from wn._core import Synset


class CstSynstet:
    def __init__(self, knowledgebase, id) -> None:
        self.id = id
        self.synset = knowledgebase.synset(id)
        self.common_ancestors = None
        self.best_common_ancestor = None
        self.children = {hypo.id for hypo in self.get_hyponyms(self.synset)}

    ##### Inside method, only interact with self

    def is_a_leaf(self):
        """return true if the candidate is a leaf in the given wordnet"""
        if len(self.children) == 0:
            return True
        else:
            return False

    def hyponyms(self):
        return self.synset.hyponyms()

    def lemmas(self):
        return self.synset.lemmas()

    def definition(self):
        return self.synset.definition()
    
    

    def get_hyponyms(self, candidate) -> set[str]:
        """Get all the hyponyms of a given synset and the hyponyms of its hyponyns ect
          until the leafs"""
        hyponyms = set()
        for hyponym in candidate.hyponyms():
            hyponyms |= set(self.get_hyponyms(hyponym))
        return hyponyms | set(self.synset.hyponyms())

    ##### Outside fondtions interct with other knowledgebase

    def is_inbetween(candidate_id, hyper_id, hypo_id, rich):
        if (
            rich.synset(hyper_id) in rich.synset(candidate_id).hypernyms()
            and rich.synset(hypo_id) in rich.synset(candidate_id).hyponyms()
        ):
            return True
        else:
            return False

    def get_Path_between():
        return None

    def find_common_ancestor_with_other_knowledge_base(self, other_kb_ids):
        """Find the common ancestor of the candidate between the two wn
        From leaf to center

        input : candidate : synset from rh
        output : - a list with all the possible common ancestors of the candidate synset
                - a list with the first common ancestor and the unknown node just before it
        """

        common_ancestor_list = []

        # { ancertr_id : { all the previous nodes}}
        last_uncommon_nodes = {}

        # so of course the candidate isn't in the current self kb as
        not_common_ancestor_list = [self.synset]

        while len(not_common_ancestor_list) > 0:
            # at the end when the list is empty it means we have all the common ancestors
            ancestor = not_common_ancestor_list[0]

            if ancestor.id in other_kb_ids:
                common_ancestor_list.append(ancestor)
                # last_uncommon_nodes.extend([
                #                                (hypo, ancestor) for hypo in ancestor.hyponyms()
                #                                    if hypo not in common_ancestor_list
                #                            ]) # keep tratck of the last uncomon node
                last_uncommon_nodes[ancestor.id] = [
                    hypo.id for hypo in ancestor.hyponyms()
                ]

                not_common_ancestor_list.remove(
                    ancestor
                )  # this ancestor is common so we remove it from the uncommun ones
            else:
                # this ancestor is not common so we add the next order ancestor and we remove the curent position
                not_common_ancestor_list.extend(ancestor.hypernyms())
                not_common_ancestor_list.remove(ancestor)

        for ancestor in last_uncommon_nodes:
            if self.id in last_uncommon_nodes[ancestor]:
                # if the ancestor is directly above the candidate we don't car about the other nodes
                # ex :  candidate is 'oewn-14178756-n' (in rh) but the common ancestor is 'oewn-14176492-n'
                # its direct hypernyms, hense  return  {'oewn-14176492-n': ['oewn-14178756-n']}) , even if
                # the hyponyms of 'oewn-14176492-n' are : ['oewn-14177098-n', 'oewn-14178756-n']
                last_uncommon_nodes[ancestor] = [self.id]

        return common_ancestor_list, last_uncommon_nodes


if __name__ == "__main__":
    ## rich = normal wn here
    rh = wn.Wordnet("oewn:2022")
    wk = wn.Wordnet("oewn-custom:2023")
    all_weak_synsets_id = [weak.id for weak in wk.synsets()]

    ## creation of the synset object ;

    frog = "oewn-01642406-n"
    rannid = "oewn-01643487-n"
    wood_frog = "oewn-01643847-n"

    synset = CstSynstet(wk, wood_frog)
    print("is wood_frog a leaf ? ", synset.is_a_leaf())
    print()
    synset = CstSynstet(rh, "oewn-14178756-n")
    print("synset 'oewn-14178756-n' wich is only present in rh : ")
    print(synset.synset.lemmas())
    print(
        "its hypernyms ",
        synset.synset.hypernyms(),
        synset.synset.hypernyms()[0].lemmas(),
    )
    print(synset.find_common_ancestor_with_other_knowledge_base(all_weak_synsets_id))

    """a = ['oewn-00101073-n' ,'oewn-00551808-n', 'oewn-06167042-n']
    for s in a : 
        print(rh.synset(s).lemmas())
    i = 0
    for sy in rh.synsets(): 
        
        if len(sy.hypernyms())>1:
            print(sy.lemmas())
            print(sy , sy.hypernyms())
            i = i+1
        if i ==4:
            break"""
