#!/usr/bin/env python3

import argparse
from typing import Any, List
import wn
from wn._core import Synset , Word

import pathlib
import datetime
import random
import json
import time

# from tqdm import tqdm

PARENT_PATH = pathlib.Path().absolute().parents[0]
SYNSET_TO_REMOVE_PATH = PARENT_PATH.joinpath("data", "synset_list")
SYNSET_TO_REMOVE_PATH_SYNSET = SYNSET_TO_REMOVE_PATH.joinpath("synsets")


class SynsetCandates:
    def __init__(self, 
                 id: str, 
                 downgrade_ratio: int, 
                 max_impact: int, 
                 file_of_already_removed_synset:str=None) -> None:
        """
        input : id              : wordnet id owen usually
                downgrade_ratio : percentage of dowgrqding of the wordnet
                max_impact      : maximun impact the removing of a synset can have, the collateral damadge of the removing of a synset
        """

        self.en = wn.Wordnet(id)
        self.synset_list = self.en.synsets()
        self.downgrade_ratio = downgrade_ratio  # the percentage of synset to remove , new_wordnet = downgrade_ratio * original_wordnet
        self.max_impact = max_impact
        self.synset2hyponyms = (
            {}
        )  ## dict wich link the synset.id to its tree, not reiniialised in the run to perform faster computation on several runs
        self.tries_synsets = []  ## list of the synset tries, reinitialised in the run
        self.synset_to_remove = set()
        self.synset_to_avoid = []

        
        if file_of_already_removed_synset is not None:
            assert ".txt" not in file_of_already_removed_synset, "No extention in the file name"
            assert ".json" not in file_of_already_removed_synset, "No extention in the file name"
        
            # read the synset list to avoid
            synset_list_tmp = []
            with open(
                SYNSET_TO_REMOVE_PATH_SYNSET.joinpath("".join([file_of_already_removed_synset, ".txt"])),
                "r",
            ) as f:
                synset_list_tmp.append(f.read())  # TODO check que ça marche
            self.synset_to_avoid = synset_list_tmp[0].split("\n")
        


    ## core methods ###########################################################

    def get_hyponyms(self, synset: Synset) -> set[str]:
        """Get all the hyponyms of a given synset and the hyponyms of its hyponyns ect until the leafs"""
        hyponyms = set()
        for hyponym in synset.hyponyms():
            hyponyms |= set(self.get_hyponyms(hyponym))
        return hyponyms | set(synset.hyponyms())

    def mesure_synset_impact(self, hyponyms_set):
        return len(hyponyms_set) / len(self.en.synsets())

    def synset_picking(self, pickup_synset:Synset) -> list[str]:
        """
        input : a candidate synset_randomly choosen Synset object
        output : list of the canditate synset and
        if the synset have a impact too important it is not removed (the impact is mesure with)
        """
        if pickup_synset.id in self.tries_synsets :
            # synset as already been tried
            # avoid double picking pick a new one
            self.synset_picking(self.random_element(self.synset_list))
        else:
            candidate_synset_to_remove_list = []
            if pickup_synset.id in self.synset2hyponyms.keys():
                # we already now the hyponyms tree of this synset thanks to a previous run
                hyponyms_set = self.synset2hyponyms[
                    pickup_synset.id
                ]  # self.synset_picking (self.random_element(self.synset_list))
            else:
                # we generate the synset tree
                hyponyms_set = self.get_hyponyms(pickup_synset)
                # we add this tree to the database
                self.synset2hyponyms[pickup_synset.id] = hyponyms_set
            if len(hyponyms_set) == 0:
                # no need to make any check in this case, its a leaf
                self.tries_synsets.append(pickup_synset.id)
                return [pickup_synset]
            else:
                if len(hyponyms_set) > self.max_impact:
                    # if the impact is to important we do the pickup process again
                    self.synset_picking(self.random_element(self.synset_list))
                else:
                    # the list of synset to remove is formed by the pickup synset and all its hyponyms
                    candidate_synset_to_remove_list.extend(
                        [pickup_synset] + list(hyponyms_set)
                    )  # extends joind the list faster
                    self.tries_synsets.extend(
                        [pickup_synset.id] + [hypo.id for hypo in hyponyms_set]
                    )
            return candidate_synset_to_remove_list

    def synset_picking_fast(self, pickup_synset:Synset)->List[Synset]:
        """
        input : a candidate synset_randomly choosen Synset object
        output : list of the canditate synset and
        if the synset have a impact too important it is not removed (the impact is mesure with)
        """

        # the list of synset to remove is formed by the pickup synset and all its hyponyms if they exists

        candidate_synset_to_remove = []
        # candidate_synset_to_remove = set()

        if pickup_synset in self.synset_to_remove or pickup_synset.id in self.synset_to_avoid :  # self.synset2hyponyms.keys() :
            # we already now the hyponyms tree of this synset thanks to a previous run
            self.synset_picking_fast(self.random_element(self.synset_list))
        else:
            # we generate the synset tree
            hyponyms_set = self.get_hyponyms(pickup_synset)
            # self.synset2hyponyms[pickup_synset.id] = [hypo.id for hypo in hyponyms_set]# TODO take into account that a hyponim file might be given
            # we add this tree to the database
            self.synset2hyponyms[pickup_synset.id] = hyponyms_set
            if len(hyponyms_set) == 0:
                # no need to make any check in this caase, its a leaf
                self.tries_synsets.append(pickup_synset)
                return [pickup_synset]
            else:
                if len(hyponyms_set) > self.max_impact:
                    # if the impact is to important we do the pickup process again
                    self.synset_picking_fast(self.random_element(self.synset_list))
                else:
                    # if an hypo has already been remooved

                    # if we work with only  list
                    # hypo_list = list(hyponyms_set-set(self.synset_to_remove))  #TODO  a way to gain time would be to transform every thing into a set
                    # candidate_synset_to_remove.extend([pickup_synset] + hypo_list ) #extends joind the list faster

                    # if we work with list and the global synset_to_remove is a set
                    candidate_synset_to_remove.extend(
                        [pickup_synset] + list(hyponyms_set)
                    )

                    # if we work with set
                    # hyponyms_set.add(pickup_synset)
                    # candidate_synset_to_remove.update(hyponyms_set)

            return candidate_synset_to_remove

    ## random picking ###########################################################
    def random_selection(self, l: list[str], number_of_sample: int) -> List[Any]:
        """output : list of randomly unique elements from l , len(output)=number_of_sample"""
        return random.sample(l, number_of_sample)

    def random_element(self, l: list[str]) -> Any:
        """output : unique element from l"""
        return random.choice(l)

    ## update parameters methods ###########################################################

    def update_downgrade_ratio(self, new_downgrade_ratio):
        self.downgrade_ratio = new_downgrade_ratio

    def update_max_impact(self, new_max_impact):
        self.max_impact = new_max_impact


#################################################
### The cut class remove one synset and cut all
### the branch starting from the removed synset
#################################################


class SynsetCandidatesCut(SynsetCandates):
    def __init__(self, id: str, downgrade_ratio: int, max_impact: int) -> None:
        super().__init__(id, downgrade_ratio, max_impact)

    # TODO

    def run(
        self,
        downgrade_ratio:int=None,
        max_impact:int=None,
        synset2hyponyms_PATH: str = None,
        filename:str=None,
    )->None:
        if downgrade_ratio is not None:
            self.downgrade_ratio = downgrade_ratio

        if max_impact is not None:
            self.max_impact = max_impact

        ## populate the list
        while len(self.synset_to_remove) < self.downgrade_ratio * len(self.synset_list):
            # while the wn hasn't been downgraded enought we continue to select sinsets
            candidate_synset_to_remove = self.synset_picking_fast(
                self.random_element(self.synset_list)
            )
            # TODO maybe see the impact of the candidate beforeadding them
            if candidate_synset_to_remove is not None:
                # set and list
                self.synset_to_remove.update(candidate_synset_to_remove)

        ## after it has been populated, reinitialisation of the tried synset for the next run   # noqa: E501
        assert self.synset_to_remove == set(self.synset_to_remove), "Not equal"

        ## save the synset list
        if filename is None:
            filename = datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S"
            )  # automatic naming

        PATH_SYNSETS = SYNSET_TO_REMOVE_PATH.joinpath("synsets")

        with open(PATH_SYNSETS.joinpath(filename + ".txt"), "w") as f:
            for row in self.synset_to_remove:
                f.write(f"{row.id}\n")  # take the id of the synset


#################################################
### The snipper class only remove one synset and
### don't cut the all branch. The hole left by the
### removed synste is then filled by lining its
### hyponyms with its hypernyms
#################################################


class SynsetCandatesSnipper(SynsetCandates):
    def __init__(self, 
                 id: str, 
                 downgrade_ratio: int, 
                 max_impact: int, 
                 file_of_already_removed_synset:str) -> None:
        super().__init__(id, downgrade_ratio, max_impact, file_of_already_removed_synset)
        self.relation_impact = {}
        # This dict correspond to the filling of the removed synset,
        # eg how we replace the missing relations
        # { Synset_2_remove { hypo : {
        #                                  Hyponym of the synset : [Hypernyms of the synset]
        #                                 }
        #                     hyper : {
        #                                  Hypernym : [All the hyponyms of the synset]
        #                                 }
        #                   }}
        # in this way we can link the hyponyms of the removes synset to its
        # hypernym. And we can deduse what to change in the hypernym

    def synset_picking_fast(self, pickup_synset:Synset)-> Any:
        """
        input : a candidate synset_randomly choosen Synset object
        output : list of the canditate synset and
        if the synset have a impact too important it is not removed (the impact is mesure with)
        """

        if pickup_synset in self.synset_to_remove or pickup_synset.id in self.synset_to_avoid:  # self.synset2hyponyms.keys() :
            # we already now the hyponyms tree of this synset thanks to a previous run
            self.synset_picking_fast(self.random_element(self.synset_list))
        else:
            # we generate the synset tree
            hyponyms_set = self.get_hyponyms(pickup_synset)
            # in this class no need to compute the hyponym tree as we
            # don't cut the all branch .
            if len(hyponyms_set) > self.max_impact:
                # if the impact is to important we do the pickup process again
                self.synset_picking_fast(self.random_element(self.synset_list))
            else:
                relation_dict = {"hypo": {}, "hyper": {}}
                for hypo in pickup_synset.hyponyms():
                    relation_dict["hypo"][hypo.id] = [
                        hyper.id for hyper in pickup_synset.hypernyms()
                    ]
                for hyper in pickup_synset.hypernyms():
                    relation_dict["hyper"][hyper.id] = [
                        hypo.id for hypo in pickup_synset.hyponyms()
                    ]

                return pickup_synset, relation_dict
        return None, None
    


    def run(
        self,
        downgrade_ratio:int=None,
        max_impact:int=None,
        synset2hyponyms_PATH: str = None,
        filename:str=None,
    )->None:
        if downgrade_ratio is not None:
            self.downgrade_ratio = downgrade_ratio

        if max_impact is not None:
            self.max_impact = max_impact

        ## populate the list
        while len(self.synset_to_remove) < self.downgrade_ratio * len(self.synset_list):
            # while the wn hasn't been downgraded enought we continue to select sinsets
            candidate_synset_to_remove, relation_dict = self.synset_picking_fast(
                self.random_element(self.synset_list)
            )

            if candidate_synset_to_remove is not None:
                self.synset_to_remove.update([candidate_synset_to_remove])
                self.relation_impact[candidate_synset_to_remove.id] = relation_dict

        ## after it has been populated, reinitialisation of the tried synset for the next run   # noqa: E501
        assert self.synset_to_remove == set(self.synset_to_remove), "Not equal"

        ## save the synset list
        if filename is None:
            filename = datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S"
            )  # automatic naming

        PATH_SYNSETS = SYNSET_TO_REMOVE_PATH.joinpath("synsets")

        with open(PATH_SYNSETS.joinpath(filename + ".txt"), "w") as f:
            for row in self.synset_to_remove:
                f.write(f"{row.id}\n")  # take the id of the synset

        ## save the json file
        if synset2hyponyms_PATH is None:
            synset2hyponyms_PATH = filename + ".json"

        PATH_RELATION = SYNSET_TO_REMOVE_PATH.joinpath("relations")

        with open(PATH_RELATION.joinpath(synset2hyponyms_PATH), "w") as outfile:
            json.dump(self.relation_impact, outfile)

def main(mode:str, 
         wn_type:str, 
         downgrade_ratio:float, 
         max_impact:int, 
         file_of_already_removed_synset:str, 
         filename:str):
    if mode == "snipper":
        print(mode, wn_type, max_impact)
        SC = SynsetCandatesSnipper(
                wn_type, 
                downgrade_ratio=downgrade_ratio, 
                max_impact=max_impact, 
                file_of_already_removed_synset = file_of_already_removed_synset
            ) 
        SC.run(downgrade_ratio=downgrade_ratio, filename=filename)
    else : 
        SC = SynsetCandidatesCut(
                    wn_type, 
                    downgrade_ratio=downgrade_ratio, 
                    max_impact=max_impact, 
                ) 
        SC.run(downgrade_ratio=downgrade_ratio, filename=filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
description="Lunch the run of the the xml_parser wich will remove synset and modify the input wordnet according to the instrucition files"
    )  
    parser.add_argument(
        "--mode", 
        "-m", 
        type=str, 
        default="snipper",
        help="Mode of the candidate generator either snipper or cut "
    )
    parser.add_argument(
        "--wn_type", 
        "-w", 
        type=str, 
        default="oewn:2022",
        help="Nome of the source wn to load (nust be in the wn database)"
    )
    parser.add_argument(
        "--downgrade_ratio",
        "-d",
        type=float,
        help="Percentage of downgrating ",
    )
    parser.add_argument(
        "--max_impact",
        "-i",
        type=int,
        default=8000,
        help="Max impact accepted",
    )
    parser.add_argument(
        "--file_of_already_removed_synset",
        "-r",
        type=str,
        help="name of the file containing the candidate synset alreading removed (no extention)",
    )
    parser.add_argument(
        "--filename", 
        "-f", 
        type=str, 
        default=None,
        help="Name to save the file "
    )

    args = parser.parse_args()

    main(args.mode, 
         args.wn_type,
         args.downgrade_ratio,
         args.max_impact,
         args.file_of_already_removed_synset, 
         args.filename)



    
    # test qu'il sont bien indep 
"""
    file_of_already_removed_synset = "20230607-140356"

    synset_list_tmp = []
    with open(
                SYNSET_TO_REMOVE_PATH_SYNSET.joinpath("".join([file_of_already_removed_synset, ".txt"])),
                "r",
            ) as f:
                synset_list_tmp.append(f.read())  # TODO check que ça marche
    synset_to_avoid = synset_list_tmp[0].split("\n")

    synset_list_tmp_2 = []
    with open(
                SYNSET_TO_REMOVE_PATH_SYNSET.joinpath("".join(["20230625-110320", ".txt"])),
                "r",
            ) as f:
                synset_list_tmp_2.append(f.read())  # TODO check que ça marche
    synset_to_avoid_2 = synset_list_tmp_2[0].split("\n")

    print(len(synset_to_avoid))
    print(len(synset_to_avoid_2))

    for s in synset_to_avoid:
        if s in synset_to_avoid_2:
            print(s)
            print("ERROR")
"""



    # else : 
    #SC = SynsetCandidatesCut(
    #            "oewn:2022", 
    #            downgrade_ratio=downgrade_ratio, 
    #            max_impact=max_impact, 
    #        ) 
    #SC.run(downgrade_ratio=downgrade_ratio)

