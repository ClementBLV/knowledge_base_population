import argparse
import json
from typing import List
import numpy as np

def compute_hits_at_k(shots:List[List[int]], element=0, k_values=[1, 3, 10]):
    """
    shots : list of ranked index (the position correspond to the index) - 
        ex : probabilities [0.9, 0.1, 0.8, 0.7] shots [0, 2, 3, 1] 
        - with 0 the élément we are interested in prosotion of the real probability 
        ~ list of list of predictions 
    element : element in the list we want to compute the hit@
    k_values : the hit@k_values
    """
    hits = {k: 0 for k in k_values}
    for ranking in shots:
        if element in ranking[: k_values[-1]]:
            for k in k_values:
                if element in ranking[:k]:
                    hits[k] += 1
    return hits

def compute_mrr(shots:List[List[int]], element=0):
    """
    Compute the Mean Reciprocal Rank (MRR) for a set of ranked results.
    
    Args:
        ranked_results: List of ranked indices for each evaluation example.
        element: The target element (e.g., the correct index) for which to compute the rank.

    Returns:
        float: The Mean Reciprocal Rank (MRR).
    """
    reciprocal_ranks = []
    for ranking in shots:
        try:
            rank_position = list(ranking).index(element) + 1
            reciprocal_ranks.append(1 / rank_position)
        except ValueError:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

