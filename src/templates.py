WN_LABELS = [
    "_hypernym",
    "_derivationally_related_form",
    "_instance_hypernym",
    "_also_see",
    "_member_meronym",
    "_synset_domain_topic_of",
    "_has_part",
    "_member_of_domain_usage",
    "_member_of_domain_region",
    "_verb_group",
    "_similar_to",
]


WN_LABEL_TEMPLATES = { # in the future we will add the indirect relations 
    "_hypernym":[
        "{obj} specifies {subj}", 
        "{subj} generalize {obj}",
    ],
    "_derivationally_related_form":[
        "{obj} derived from {subj}", 
        "{subj} derived from {obj}",
    ],
    "_instance_hypernym":[
        "{obj} is a {subj}",
        "{subj} such as {obj}",
    ],
    "_also_see":[
        "{obj} is seen in {subj}",
        "{subj} has 		 {obj}",
    ],
    "_member_meronym":[ 
        "{obj} is the family of {subj}",
        "{subj} is a member of {obj}",
    ],
    "_synset_domain_topic_of":[
        "{obj} is a topic of {subj}",
        "{subj} is the context of {obj}",
    ],
    "_has_part":[  
        "{obj} contains {subj}",
        "{subj} is a part of {obj}",
    ],
    "_member_of_domain_usage":[ 
        "{obj} is a domain usage of {subj}",
        "X",
    ],
    "_member_of_domain_region":[ 
        "{obj} is the domain region of {subj}",
        "{subj} belong to the regieon of {obj}",
    ],
    "_verb_group":[  
        "{obj} is synonym to {subj}",
        "{subj} is synonym to {obj}",
    ],
    "_similar_to":[ 
        "{obj} is similar to {subj}",
        "{subj} similar to {obj}",
    ],
}


FORBIDDEN_MIX = {
    "_hypernym":[
        "{obj} derived from {subj}",
        "{subj} derived from {obj}",        
        "{obj} is a {subj}", 
        "{subj} such as {obj}",
        ],
    "_derivationally_related_form":[
        "{obj} specifies {subj}", 
        "{subj} generalize {obj}",
        "{obj} is a {subj}", 
        "{subj} such as {obj}",
        ],
    "_instance_hypernym":[
        "{obj} specifies {subj}", 
        "{subj} generalize {obj}",
        "{obj} derived from {subj}",
        "{subj} derived from {obj}",    
    ],
}

templates_direct = [
    "{obj} specifies {subj}",
    "{obj} derived from {subj}",
    "{obj} is a {subj}",
    "{obj} is seen in {subj}",
    "{obj} is the family of {subj}",
    "{obj} is a topic of {subj}",
    "{obj} contains {subj}",
    "{obj} ?????????? {subj}",
    "{obj} is the domain region of {subj}",
    "{obj} is synonym to {subj}", 
    "{obj} is similar to {subj}" 
]

template_indirect = [
    "{subj} generalize {obj}",
    "{	X }",
    "{subj} such as {obj}",
    "{subj} has 		 {obj}",
    "{subj} is a member of {obj}",
    "{subj} is the context of {obj}",
    "{subj} is a part of {obj}",
    "{obj} ?????????? 	    {subj}"
    "{subj} belong to the regieon of {obj}",
    "{subj} is synonym to {obj}",
    "{subj} similar to {obj}",
]
