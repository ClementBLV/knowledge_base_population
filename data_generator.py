import random 
import wn

class Datagen : 
    def __init__(self): 
        en = wn.Wordnet('oewn:2022')
        self.synsets = en.synsets()

    def get_n_random_synset(self, synset,  n ):
        s = random.choices(self.synsets, k=n)
        while synset in s : 
            s = random.choices(self.synsets, k=n)
        return s

    def get_firsy_order_hypos(self,synset ):
        first_order_hypo =  synset.hyponyms()
        return first_order_hypo

        
    def get_second_order_hypos(self, synset ):
        second_order_hypo = []
        for h in self.get_firsy_order_hypos(synset ) : 
            second_order_hypo.extend(h.hyponyms())
        return second_order_hypo

    def get_first_order_hyper(self, synset):
        first_order_hyper =  synset.hypernyms()
        return first_order_hyper


    def get_second_order_hyper(self, synset):
        second_order_hyper = []
        for h in self.get_first_order_hyper(synset) : 
            second_order_hyper.extend(h.hypernyms())
        return second_order_hyper
    
    def get_third_order_hyper(self, synset):
        third_order_hyper = []
        for h in self.get_second_order_hyper(synset) : 
            third_order_hyper.extend(h.hypernyms())
        return third_order_hyper


    def get_test_set_synset_4entailment (self, synset, n , m, order = 1, hyper = False ):
        """ return { False : n random synset 
                    True  : the hyponym of the synset }
                    
                    hyper = False to do with the hypernym relatipn 
                    """

        hypos = list(synset.hyponyms())
        hypers = list(synset.hypernyms())
        
        if not(hyper) and order == 1 : 
            true_set = random.choices(hypos, k= m )
        if not(hyper) and order == 2 : 
            true_set = random.choices(hypos+self.get_second_order_hypos(synset), k= m )


        if hyper and order == 1 : 
            true_set = random.choices(hypers, k= m )
        if hyper and order == 2 : 
            true_set = random.choicse(hypers+ self.get_second_order_hyper(synset ), k= m )
    

        return {'Synset' : synset , False: self.get_n_random_synset(synset,  n ), True : true_set}


    def verbalise (self, test_set_synset): 
        """ Directly verbilise the dictionary obtained by get_test_set_synset into a dictionnary where the synsets 
            are replace by tuples of all there lemmas and definitions """
        verbalised_dict = {'Synset':test_set_synset['Synset']  , False : [], True : []}
        for synset in test_set_synset[False]:
            for lemma in synset.lemmas():
                verbalised_dict[False].append((lemma , synset.definition()))
        
        for hypo in test_set_synset[True]:
            for lemma in hypo.lemmas():
                verbalised_dict[True].append((lemma , hypo.definition()))
        return verbalised_dict
    

    ##### Similarity methods 

    def get_path_hypernyms (self, synset , ancestror ):
        # get the path of hypernym starting from the synset to the ancestor  
        curent_ancestor = synset.hypernyms()[0]
        ancestor_list = [synset, curent_ancestor]
        i = 0
        while curent_ancestor != ancestror : 
            i = i + 1
            assert i < 10 , "To many steep betweens candidates"
            curent_ancestor = curent_ancestor.hypernyms()[0]
            ancestor_list.append(curent_ancestor)
        return(ancestor_list)

    def get_one_path_hyponyms (self, synset ): 
        # get a rambomly choosen path hyponym of hyponyms strating from the target synset 
        curent_decendance_list = synset.hyponyms()
        if curent_decendance_list == []:
            return []
        else : 
            current_descendance = random.choice(curent_decendance_list)
            if current_descendance.hyponyms() == []:
                return [current_descendance]
            else : 
                return [current_descendance]+[random.choice(current_descendance.hyponyms())]
            
    def get_one_full_path (self, synset , ancestror ): 
        hypernyms  = [h for h in reversed(self.get_path_hypernyms (synset , ancestror ) )]
        hyponyms = self.get_one_path_hyponyms (synset )
        return (hypernyms+hyponyms)

    def get_one_wrong_path (self, good_path, synset , impact = 0):
        """ get a wrong path with the same origin as the candidate synset but with 
        a biffurcation at the impact point if impact = 0 the synset just above the candidate
        and if impact = 1 the  2 orer hypernym; ex with impact = 1 
            Candidate --------------
            ['salamander']
            True -------------------
            ['craniate', 'vertebrate']
            ['amphibian']
            ['salamander']
            ['Dicamptodon ensatus', 'Pacific giant salamander']
            False -------------------
            ['craniate', 'vertebrate']
            ['amphibian']
            ['blindworm', 'caecilian']"""
        if good_path [1] == synset or impact ==1: 
            new_start = random.choice(good_path[0].hyponyms())
            i = 0
            while new_start == synset and i <10 : 
                new_start = random.choice(good_path[0].hyponyms())
            new_path = self.get_one_full_path (new_start , good_path [0] )
            return new_path
        else :  
            new_start = random.choice(good_path[1].hyponyms())
            new_path = self.get_one_full_path (new_start , good_path [0] )
            #if synset in new_path : 
            #    self.get_one_wrong_path (good_path, synset , impact = impact) # break if the synset is a leaf or if there is only one path possible 
            return self.get_one_full_path (new_start , good_path [0] )
        
    def get_test_set_synset_4similarity (self, synset , ancestror, impact, rand = False, no_hyponyms = False):
        """ return a dictionnary with the forme 
        { Synset : candidate synset , 
        True : [ancestor, h , h' , h'' ...], 
        False :[ancestor , h' , bifurcation ] 
            or it can be a total random synset is rand parameter == True""" 
        if no_hyponyms:
            good_path = [h for h in reversed(self.get_path_hypernyms (synset , ancestror ))]
        else :    
            good_path = self.get_one_full_path (synset , ancestror )
        if rand : 
            random_synset = random.choice(self.synsets) 
            while random_synset.hypernyms()==[]:
                random_synset = random.choice(self.synsets) 
            random_ancestor = random.choice(random_synset.hypernyms())
            if random_ancestor.hypernyms() != []:
                random_ancestor_of_ancestor = random.choice(random_ancestor.hypernyms())
                wrong_path =  self.get_one_full_path (random_synset , random_ancestor_of_ancestor)
            else : 
                wrong_path =  self.get_one_full_path (random_synset , random_ancestor )
        else :       
            wrong_path = self.get_one_wrong_path (good_path, synset , impact = impact)
        return {'Synset': synset , True : good_path , False : wrong_path}

    
