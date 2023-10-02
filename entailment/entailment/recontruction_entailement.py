from typing import Any, List, Tuple

try:
    from entailement.Entailement import Model_loader, Tokenizer
    from entailement.SynsetConvertor import SynsetConvert
except:
    from Entailement import Model_loader, Tokenizer
    from SynsetConvertor import SynsetConvert

from collections import Counter
from wn._core import Synset
import numpy as np


class Entail:
    def __init__(
        self, path_hyper: str, path_hypo: str, index_hyper: int, index_hypo: int
    ) -> None:
        assert path_hyper is None and path_hypo is not None, "Both path must be given "
        assert path_hyper is not None and path_hypo is None, "Both path must be given "

        if not (path_hyper is None and path_hypo is None):
            self.model = Model_loader(
                hypo_path=path_hyper,
                hyper_path=path_hypo,
            )
        ## the index is mainly used for the evaluation script
        try:
            self.model = Model_loader(
                hypo_path="entailement/model_hypo_epoch_" + str(index_hypo) + ".pth",
                hyper_path="entailement/model_hyper_epoch_" + str(index_hyper) + ".pth",
            )
        except:
            self.model = Model_loader(
                hypo_path="model_hyper_epoch_" + str(index_hypo) + ".pth",
                hyper_path="model_hyper_epoch_" + str(index_hyper) + ".pth",
            )

        self.tokenizer = Tokenizer()

    def most_frequent(self, l: List[Any]) -> Any:
        """Return the most frequent item"""
        occurence_count = Counter(l)
        return occurence_count.most_common(1)[0][0]

    def cathegorized_synset(
        self, candidate_synset: Synset, synset_to_cathegorized: Synset
    ) -> str:
        """Find the relation between two synsets, the relation can be either
        HYPERNYM , HYPONYM or None if the outputs or the model contradict each other and
        can't be interpreted

        Args:
            candidate_synset (Synset): synset candidate, a synset object
            synset_to_cathegorized (wn.Synset): synset target from the branch

        Returns:
            str : of the relation between the two synsets
        """

        # initialisation
        relation_hyper, relation_hypo = None, None

        #### HYPERNYM ####

        ## check if target is hyponyms : target generalize candidate AND candidate specifies target
        # target generalize candidate

        input = SynsetConvert(
            candidate_synset=candidate_synset,
            target_synset=synset_to_cathegorized,
            type="hyper",
        ).input  # candidate specifies target
        input_tokenized = self.tokenizer.tokenize(
            hypothesis=[input["hypothesis"]], premise=[input["premise"]]
        )
        prediction1 = self.model.predict(input_tokenized, model_type="hyper")

        # candidate specifies target
        input = SynsetConvert(
            candidate_synset=synset_to_cathegorized,
            target_synset=candidate_synset,
            type="hypo",
        ).input
        input_tokenized = self.tokenizer.tokenize(
            hypothesis=[input["hypothesis"]], premise=[input["premise"]]
        )
        prediction2 = self.model.predict(input_tokenized, model_type="hypo")

        # interpretation
        if prediction1 == [0] and prediction2 == [0]:
            relation_hyper = "HYPERNYM"

        #### HYPONYM ####

        ## check if target is hypernyms : target specifies candidate AND candidate generalise candidate
        # target generalize candidate
        input = SynsetConvert(
            candidate_synset=candidate_synset,  # ordered inversed compared to before
            target_synset=synset_to_cathegorized,
            type="hypo",
        ).input  # candidate specifies target
        input_tokenized = self.tokenizer.tokenize(
            hypothesis=[input["hypothesis"]], premise=[input["premise"]]
        )
        prediction1 = self.model.predict(input_tokenized, model_type="hypo")

        # candidate specifies target
        input = SynsetConvert(
            candidate_synset=synset_to_cathegorized,
            target_synset=candidate_synset,
            type="hyper",
        ).input
        input_tokenized = self.tokenizer.tokenize(
            hypothesis=[input["hypothesis"]], premise=[input["premise"]]
        )
        prediction2 = self.model.predict(input_tokenized, model_type="hyper")

        # interpretation
        if prediction1 == [0] and prediction2 == [0]:
            relation_hypo = "HYPONYM"

        ### FUSION OF THE RESULTS
        if (relation_hyper is None and relation_hypo is None) or (
            relation_hyper is not None and relation_hypo is not None
        ):  # if both relation can't be entail then its none
            return None

        if relation_hypo is not None:
            return "HYPONYM"

        if relation_hyper is not None:
            return "HYPERNYM"

    def cathegorize_branch(
        self, candidate_synset: Synset, branch: np.ndarray
    ) -> List[str]:
        """Convert the branch into each relations,
        ex ['HYPERNYM', 'HYPONYM', None] for a branch of three
        synset.

        Args:
            candidate_synset (Synset): synset candidate, a synset object
            branch (np.ndarray): one branch of synset outputed from the reconstuction

        Returns:
            List[str]: the list of all the relation wich link the synsets of the
            branch to the candidate
        """
        branch_info = []
        for synset in branch:
            if synset is not None:
                # print(synset)
                branch_info.append(self.cathegorized_synset(candidate_synset, synset))
        return branch_info

    def cathegorize_output(
        self, candidate_synset: Synset, branches_scores: Tuple
    ) -> np.ndarray:
        """Insert the candidate synset at the right position in the branches given in imput

        Args:
            candidate_synset (Synset): candidate synset
            branches_scores (Tuple(np.ndarray, List[int])): Output from the recontruction process, hence a tuple
            formed buy the matrix of branches and their respective scores

        Returns:
            np.ndarray: Matrix with the candidate synset located at the right position
        """
        # INITIALISATION
        cancidate_hyper_index = []
        cancidate_hypo_index = []
        scores = branches_scores[1]
        patate_condition = False

        # Loop though the branches results
        for index, branch in enumerate(branches_scores[0]):
            if scores[index] == 1:
                sure_branch = True
            else:
                sure_branch = False

            # change all the synsets of the branch by its relation
            cathegorized_branch = self.cathegorize_branch(candidate_synset, branch)

            index_hypo = None
            index_hyper = 0  # initialized to the common ancestor position

            while (
                index_hyper < len(branch)
                and cathegorized_branch[index_hyper] == "HYPERNYM"
            ):
                # index is the last occurence of the HYPERNYM relation
                index_hyper = index_hyper + 1
            if index_hyper > 0:
                index_hyper = index_hyper - 1
                # if index is still 0 that mean no HYPERNYM hense it stay to zero

            for index, relation in enumerate(cathegorized_branch):
                # index hypo is the first occurence of the HYPONYM relation
                if relation == "HYPONYM":
                    index_hypo = index
                    break

            if index_hypo is not None:
                if index_hyper == index_hypo - 1 or scores[index] == 1:
                    # the hypernym as the hyponyms are right after another
                    # the candidate is inbetween
                    cancidate_hyper_index.append(index_hyper)
                    cancidate_hypo_index.append(index_hypo)
                else:
                    # case HYPERNYM - None - HYPONYM
                    cancidate_hyper_index.append(index_hyper)
                    cancidate_hypo_index.append(100000)
                    # 100000 is an arbitrary index to precise that the candidate
                    # is identified as a leaf and don't have any valid hyponyms

            elif (index_hypo is None and index_hyper is not None) or (
                index_hyper is not None and index_hypo != index_hyper - 1
            ):
                # the candidate don't have hyponyms and its a leaf of the hypernym
                cancidate_hyper_index.append(index_hyper)
                cancidate_hypo_index.append(100000)  # to signify no hyponyms

            else:
                patate_condition = True
                # we can't conclude I don't know what to do
                # we plug it to the clossest common ancestor
                cancidate_hyper_index.append(0)
                cancidate_hypo_index.append(100000)  # to signify no hyponyms

        ## take the most common index of hypernyms and hyponyms
        most_frequent_hyper = self.most_frequent(cancidate_hyper_index)
        most_frequent_hypo = self.most_frequent(cancidate_hypo_index)

        ## reconstruct the output matrix
        new_matrix = []
        new_branch = []
        # go through all the branch and modify them according to the position of
        # the synset to insert
        for index, branch in enumerate(branches_scores[0]):
            new_branch = branch.tolist()[: most_frequent_hyper + 1]
            new_branch.append(candidate_synset)

            if cancidate_hypo_index[index] == 100000:
                patate_condition = True
                new_branch.extend(
                    [None for i in range(0, len(branch[most_frequent_hyper:]) - 1)]
                )  # padding with none for the case where there are no hyponyms
            else:
                new_branch.extend(branch[most_frequent_hyper + 1 :].tolist())
            new_matrix.append(new_branch)

        return np.array(new_matrix), patate_condition


if __name__ == "__main__":
    import wn

    en = wn.Wordnet("oewn:2022")

    candidate = en.synsets("republic")[0]
    target = candidate.hypernyms()[0]
    print(candidate.lemmas())
    print(target.lemmas())
    # print(cathegorized_synset(candidate, target))
