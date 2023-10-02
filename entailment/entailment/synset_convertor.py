#!/usr/bin/env python3
from pprint import pprint
class SynsetConvert:
    def __init__(self, candidate_synset, target_synset, type):
        self.candidate_synset = candidate_synset
        self.target_synset = target_synset

        self.premices_dict = {"hyper": " generalize ", "hypo": " specifies "}

        self.synset_dict = {}
        self.synset_dict["candidate"] = {
            "id ": self.candidate_synset.id,
            "lemmas": self.candidate_synset.lemmas(),
            "definition": self.candidate_synset.definition(),
        }

        if type == "hypo":
            self.synset_dict["hypo"] = {
                "id ": self.target_synset,
                "lemmas": self.target_synset.lemmas(),
                "definition": self.target_synset.definition(),
            }
        if type == "hyper":
            self.synset_dict["hyper"] = {
                "id ": self.target_synset,
                "lemmas": self.target_synset.lemmas(),
                "definition": self.target_synset.definition(),
            }

        self.input = self.transformater(self.synset_dict, type)
        


    def fuse_context(self, elt_dict):
        if elt_dict["definition"][0] == "a":
            return " ".join(elt_dict["lemmas"]) + " is " + elt_dict["definition"]
        return " ".join(elt_dict["lemmas"]) + " is a " + elt_dict["definition"]

    def transformater(self, candidate_dict: dict, type: str) -> dict:
        """Convert the input dictionnary into a sentense

        Args:
            candidate_dict (dict): one candidate dictionnary of the candidate dataset
            key (str): key in the dictionnary

        Returns:
            dict: { hypohesis : fused context
                    hypernym :
                    hyponym :     }
        """
        assert type in ["hyper", "hypo"], "Type is either hypernym or hyponym"
        transformed_dict = {
            "hypothesis": self.fuse_context(candidate_dict[type])
            + ". "
            + self.fuse_context(candidate_dict["candidate"]),

            "premise": " ".join(candidate_dict[type]["lemmas"])
            + self.premices_dict[type]
            + " ".join(candidate_dict["candidate"]["lemmas"]),
        }

        return transformed_dict
