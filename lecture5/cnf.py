"""Encoding "exactly k of these variables are true" as CNF clauses."""
from itertools import combinations


def at_most(k, variables):
    # No k+1 of them can all be true.
    return [[-v for v in combo] for combo in combinations(variables, k + 1)]


def at_least(k, variables):
    # Every choice of n-k+1 of them must contain a true one.
    return [list(combo)
            for combo in combinations(variables, len(variables) - k + 1)]


def exactly(k, variables):
    return at_most(k, variables) + at_least(k, variables)
