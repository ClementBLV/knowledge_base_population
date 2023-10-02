#!/usr/bin/env python3


# Library import
from lxml import etree

from tqdm import tqdm
import pathlib
import datetime
import json
import os
import argparse
import time

# from bs4 import BeautifulSoup
# import lxml.etree as et
# from pprint import pprint
# import lxml
# import cchardet

# path to work with
PARENT_PATH = pathlib.Path().absolute().parents[0]
XML_PATH = PARENT_PATH.joinpath("data", "source")
XML_SAVE_PATH = PARENT_PATH.joinpath("data", "generated")
CONFIG_FILE_PATH = PARENT_PATH.joinpath("src", "config.json")
SYNSET_TO_REMOVE_PATH = PARENT_PATH.joinpath("data", "synset_list")
SYNSET_TO_REMOVE_PATH_SYNSET = SYNSET_TO_REMOVE_PATH.joinpath("synsets")
SYNSET_TO_REMOVE_PATH_RELATIONS = SYNSET_TO_REMOVE_PATH.joinpath("relations")

# in synset list there are a file synset for
# the synset to individually erased from the
# database and there can also have a relation
# json file with the same name to indicates
# the relation to changed .


class XMLParser:
    def __init__(self, word_net_path, lxml_data=None) -> None:
        """inputs :
        word_net_path    : file name of the xml
        """
        if word_net_path == "oewn":
            word_net_path = "english-wordnet-2022.xml"

        assert ".xml" in word_net_path, "File must be an XML"

        if lxml_data is None:
            # Reading the data inside the xml
            # file to a variable under the name
            # data
            with open(XML_PATH.joinpath(word_net_path), "r") as f:
                data = f.read()
            data = [d for d in data if d != "\n"]

            # Passing the stored data inside
            # the beautifulsoup parser, storing
            data = "".join(data)

            # self.Bs_data = BeautifulSoup(data, "xml")
            self.lxml_data = etree.fromstring(data.encode("utf-8"))

        else:
            # self.Bs_data = Bs_data
            self.lxml_data = lxml_data
        self.synset_to_remove = []

    def run(self, file_name_synset, file_name_to_save=None):
        """the method can be run on different synset files, the hyponyms list is
          concerve so it should be fatser
        inputs :
           file_name_synset : file name of the synset list to remove without any
                              extention

        """
        assert ".txt" not in file_name_synset, "No extention in the file name"
        assert ".json" not in file_name_synset, "No extention in the file name"
        if file_name_to_save is not None:
            assert (
                "xml" in file_name_to_save
            ), "No extention the file name to save must be xml"  # noqa: E501

        # read the synset list to remove
        synset_list_tmp = []
        with open(
            SYNSET_TO_REMOVE_PATH_SYNSET.joinpath("".join([file_name_synset, ".txt"])),
            "r",
        ) as f:
            synset_list_tmp.append(f.read())  # TODO check que ça marche
        self.synset_to_remove = synset_list_tmp[0].split("\n")

        # see if we are in snipper mode and some relation file have been generated
        condition = True  # if condition they are relations
        try:
            f = open(
                SYNSET_TO_REMOVE_PATH_RELATIONS.joinpath(
                    "".join([file_name_synset, ".json"])
                ),
                "r",
            )
            relations_dict = json.load(f)
        except Exception as err:
            print("\t", Exception, err, "\n")
            condition = False

        # Update
        for synset_key in tqdm(
            self.synset_to_remove
        ):  # TODO put plurial in synsets
            if len(synset_key) > 0:
                # Update relations
                if condition:
                    self.add_relation(relations_dict[synset_key])
                # Update synset :
                self.remove_synset(synset_key)

                self.remove_synset_from_relations(synset_key)

                self.remove_synset_from_word(synset_key)

        # Update the config
        self.config_update()

        # Save
        if file_name_to_save is None:
            # automatic naming
            file_name_to_save = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            file_name_to_save = "".join([file_name_to_save, ".xml"])
        self.save(XML_SAVE_PATH.joinpath(file_name_to_save))
        print("File saved in " + str(XML_SAVE_PATH.joinpath(file_name_to_save)))

    ## GET METHODS #####################################################################
    def get_word(self, word_key) -> list:
        """Input the key of a word
        TODO also be able to input the word without a key"""
        for word in self.lxml_data.findall(".//LexicalEntry"):  # senses :
            # a loop cause they might have several senses in a word
            if word.attrib["id"] == word_key:
                return word

    def get_synset(self, synset_key) -> list:
        # for synset in self.Bs_data.find_all('Synset'):
        #    if synset['id'] == synset_key:
        #        return synset
        for synset in self.lxml_data.findall(".//Synset"):  # senses :
            # a loop cause they might have several senses in a word
            if synset.attrib["id"] == synset_key:
                return synset

    def synset_to_word(self, synset_key: str) -> list:
        """Get the word link to the synset given in the key
        inputs :
            synset_key : the key of the synset (its id )
        return :
            list of LexicalEntry conneted to the synset given in imput
        <LexicalEntry id="oewn-100-a">
            <Lemma partOfSpeech="a" writtenForm="100" />
            <Sense id="oewn-100__5.00.00.cardinal.00" synset="oewn-02203776-s" />
        </LexicalEntry>
        """
        lexicalEntries = []
        for sense in self.lxml_data.findall(".//Sense"):  # senses :
            # a loop cause they might have several senses in a word
            if sense.attrib["synset"] == synset_key:
                # the synset in the sense we append the parent eg the lexical entry to the lexical entry
                lexicalEntries.append(sense.getparent())
        return lexicalEntries

    ## ADD METHODS #####################################################################

    def add_relation(self, relations):
        """
          Update the relation in case of the snyper case, we link the hyponyms of the
          removed synset to its hypernyms and vis-versa
          input : relations = { hypo : {
        #                                    Hyponym : [All the hypernym of the synset]
        #                                 }
        #                     hyper : {
        #                                  hyper_id : [hypo_id , hypo_id ....]
        #                                  Hypernym : [All the hyponyms of the synset]
        #                                 }
        #                   }

        if the relation dict look like that :
        #   {'hypo': {}, 'hyper': {'oewn-03341784-n': []}}
        that means the removed synset was already a leaf hence it didn't have any hyponyms
        that's why the hypo dict is empty cause there aren't any hypo id, same th hypernym
        can't have any hyponyms cause it has now become a leaf

        !!! WE DON'T REMOVE THE EXITING RELATION WITH THE SYNSET_KEY AS IT WILL BE DONE
        IN THE NEXT STEP OF THE RUN !!!!
        So we add a new relation into this synset

          <Synset id="synset_key" ili="i53919" lexfile="noun.artifact" members="oewn-flying_buttress-n oewn-arc-boutant-n" partOfSpeech="n">
              <Definition>a buttress that stands apart from the main structure and connected to it by an arch</Definition>
              <SynsetRelation relType="hypernym" target="hyper_key" />
              <SynsetRelation relType="hyponym" target="hypo_key" />
          </Synset>
        """  # noqa: E501
        if len(relations["hypo"]) > 0:
            # case where there aren't any hypo cause the synset removed was already a leaf
            # so we don't have to create anything
            # acces the hypernym
            for hyper_id in relations["hyper"]:
                # find the hyper thanks to its id synset object
                hyper = self.get_synset(hyper_id)  # cause a synset is unique
                # for hyper in hyper_list:
                # see the list of hyponyms of the hyper id
                if hyper is not None:
                    for hypo_id in relations["hyper"][hyper_id]:
                        # hyper
                        # <Synset id="hyper_id"
                        #   <SynsetRelation relType="hyponym" target="hypo_id" />

                        ## create the new tag
                        subtag2 = etree.Element(
                            "SynsetRelation", relType="hyponym", target=hypo_id
                        )

                        # insert in the synset
                        hyper.insert(1, subtag2)

            # acces the hyponym
            for hypo_id in relations["hypo"]:
                # find the hypo thanks to its id
                hypo = self.get_synset(hypo_id)
                # for hypo in hypo_list:
                if hypo is not None:
                    # get all the hyper_id of this hypo
                    for hyper_id in relations["hypo"][hypo_id]:
                        # <SynsetRelation relType="hypernym" target="hyper_id" />
                        ## create the new tag
                        subtag2 = etree.Element(
                            "SynsetRelation", relType="hypernym", target=hyper_id
                        )

                        # insert in the synset
                        hypo.insert(1, subtag2)

    ## REMOVES METHODS #################################################################

    # WORD
    def remove_word(self, word_key: str):
        """Void Method : Only remove the LexicalEntry(word) ententy from the dataBase."""
        # self.Bs_data.find_all("LexicalEntry ",{'id':word_key}) -> [] be CAREFULL NO ""
        word = self.get_word(word_key)
        word.getparent().remove(word)  # remove the LexicalEntry(word) elenent

    # SYNSET

    # to remove a synset, you should remove its prencence,
    # as a synset, in relation between synsets and in referencein words
    def remove_synset(self, synset_key: str):
        """Void Method : Only remove the synset ententy from the dataBase. It doesn't removes the references"""
        # synset_list = self.Bs_data.find_all('Synset',{'id':synset_key}) # nos its a simple find
        synset = self.get_synset(synset_key)
        synset.getparent().remove(synset)  # remove the synset elenent

    def remove_synset_from_relations(
        self,
        synset_key: str,
    ):
        """Remove the relation liked to the symset_key
        inputs :
            synset_key : the key of the synset (its id )

        """
        # initialisation
        # takeaways = self.Bs_data.findAll('Synset') # all the words of the DataBase
        # loops
        # for eachtakeaway in takeaways:
        #        for relation in eachtakeaway.find_all('SynsetRelation'):  # a loop cause they might have several senses in a word
        # relations = self.Bs_data.findAll('SynsetRelation')
        for relation in self.lxml_data.findall(
            ".//SynsetRelation"
        ):  # gain of 0.9s on the programm and less variable stokck in ram
            if relation is not None:
                synset = relation.attrib[
                    "target"
                ]  # a sense refer to a synset, a word might be in several synsets cause it has different senses
                if synset == synset_key:
                    relation.getparent().remove(relation)

    def remove_sense_from_relation(self, sense_key: str):
        """The sent relation or the relation between sense when we removes a sense
        we should also remove its refrenses it the other sense raltions  :

          <LexicalEntry id="oewn-acquisitive-a">
              <Lemma partOfSpeech="a" writtenForm="acquisitive" />
              <Sense id="oewn-acquisitive__3.00.00.." synset="oewn-00029456-a">
                  <SenseRelation relType="antonym" target="oewn-unacquisitive__3.00.00.." />
                  <SenseRelation relType="derivation" target="oewn-acquire__2.40.00.." />
                  <SenseRelation relType="derivation" target="oewn-acquisitiveness__1.07.00.." />
              </Sense>
          </LexicalEntry>
        """
        for sense_relation in self.lxml_data.findall(
            ".//SenseRelation"
        ):  # gain of 0.9s on the programm and less variable stokck in ram
            if sense_relation is not None:
                sense = sense_relation.attrib[
                    "target"
                ]  # a sense refer to a synset, a word might be in several synsets cause it has different senses
                if sense == sense_key:
                    sense_relation.getparent().remove(sense_relation)
        # TODO add it to the test

    """If the word have only one synset and we remove this synset then we also remove the word. 
    If the word have several synsets (eg senses), we only remove the sense and keep the word"""

    def remove_synset_from_word(
        self,
        synset_key: str,
    ):
        """Remove the synset inside a word related to the given in the symset_key
        inputs :
            Bs_data : teh beatifulsoup dataBase from the xml
            synset_key : the key of the synset (its id )

        """

        # initialisation
        senses = self.lxml_data.findall(".//Sense")

        # loops
        for (
            sense
        ) in (
            senses
        ):  # senses :  # a loop cause they might have several senses in a word
            if sense is not None:
                # a sense refer to a synset, a word might be in several synsets cause it has different senses
                if sense.attrib["synset"] == synset_key:
                    sense_parent = sense.getparent()  # a lexical entry
                    # removes the sense from the other relation where it appears
                    self.remove_sense_from_relation(sense.attrib["id"])

                    if len(sense_parent.findall(".//Sense")) == 1:
                        # only one sense to remove so we get rid of all the parent
                        sense_parent.getparent().remove(sense_parent)
                    else:
                        sense_parent.remove(sense)

    # UPDATE THE FILE CONFIG ############################################################################

    def config_update(self):
        """Change the xml config according to the config file given in imput"""
        f = open(CONFIG_FILE_PATH)
        config = json.load(f)
        for key, value in config.items():
            # self.Bs_data.find('Lexicon')[key] = os.path.join(value)
            self.lxml_data.find(".//Lexicon").attrib[key] = os.path.join(value)

    # SAVE METHODS #################################################################################################

    def save(self, xml_name):
        assert ".xml" in str(xml_name), "The file must be an xml "
        # with open(XML_SAVE_PATH.joinpath(xml_name), 'wb') as f:
        #    f.write(self.Bs_data.prettify("UTF-8"))
        # cause in wn library it check if its in "UTF-8" not uft-8
        tree = self.lxml_data.getroottree()

        tree.write(
            XML_SAVE_PATH.joinpath(xml_name),
            pretty_print=True,
            xml_declaration=True,
            encoding="UTF-8",
        )


# MAIN


def main(word_net_path, file_name_synset, file_name_to_save):
    XMLP = XMLParser(word_net_path)
    XMLP.run(file_name_synset, file_name_to_save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lunch the run of the the xml_parser wich will remove synset and modify the input wordnet according to the instrucition files"
    )  
    parser.add_argument(
        "--word_net_path", 
        "-w", 
        type=str, 
        help="path of the wordnet to modify"
    )
    parser.add_argument(
        "--file_name_synset",
        "-n",
        type=str,
        help="name of the file containing the candidate synset without extention",
    )
    parser.add_argument(
        "--file_name_to_save",
        "-s",
        default=None,
        type=str,
        help="name of the file containing the candidate synset ",
    )

    args = parser.parse_args()

    main(args.word_net_path, args.file_name_synset, args.file_name_to_save)
