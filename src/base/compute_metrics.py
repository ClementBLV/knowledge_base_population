import argparse
import json
import numpy as np

def compute_hits_at_k(probabilities, element=0, k_values=[1, 3, 10]):
    """
    shots : list of ranked index (the position correspond to the index)
    element : element in the list we want to compute the hit@
    k_values : the hit@k_values
    """
    hits = {k: 0 for k in k_values}
    for ranking in probabilities:
        if element in ranking[: k_values[-1]]:
            for k in k_values:
                if element in ranking[:k]:
                    hits[k] += 1
    return hits

def compute_mrr(probabilities, element=0):
    """
    Compute the Mean Reciprocal Rank (MRR) for a set of ranked results.
    
    Args:
        ranked_results: List of ranked indices for each evaluation example.
        element: The target element (e.g., the correct index) for which to compute the rank.

    Returns:
        float: The Mean Reciprocal Rank (MRR).
    """
    reciprocal_ranks = []
    for ranking in probabilities:
        try:
            rank_position = list(ranking).index(element) + 1
            reciprocal_ranks.append(1 / rank_position)
        except ValueError:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

