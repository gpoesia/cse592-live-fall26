"""DPLL SAT solver.

A formula is a list of clauses, a clause is a list of integer literals:
7 means "variable 7 is true", -7 means "variable 7 is false".

Example. Suppose we have 3 propositions and use P := 1, Q := 2, R := 3.

The formula: (P or Q or not R) and (not P or not Q) would be:

[
    [1, 2, -3],
    [-1, -2]
]
"""


def simplify(clauses: list[list[int]], lit) -> list[list[int]]:
    # Delete all the clauses where lit appears
    # Delete -lit from the remaining clauses.
    new_clauses = []
    for c in clauses:
        if lit in c:
            continue
        new_clauses.append([l for l in c if l != -lit])
    return new_clauses


def unit_literal(clauses) -> int | None:
    "Find a unit literal."
    for c in clauses:
        if len(c) == 1:
            return c[0]
    return None

def pure_literal(clauses) -> int | None:
    seen = set([l for c in clauses for l in c])
    for lit in seen:
        if -lit not in seen:
            return lit
    return None


def dpll(clauses, assignment) -> dict[int, bool] | None:
    """Return a satisfying assignment {variable: True/False}, or None."""
    dpll.calls += 1

    # Base cases
    if [] in clauses:
        return None  # UNSAT: empty clause

    if clauses == []:
        return assignment  # SAT! No more clauses

    # Unit propagation
    l = unit_literal(clauses)
    if l is not None:
        dpll.propagations += 1
        return dpll(
            simplify(clauses, l),
            {**assignment, abs(l): l > 0}
        )

    # Pure literal
    l = pure_literal(clauses)
    if l is not None:
        dpll.pures += 1
        return dpll(
            simplify(clauses, l),
            {**assignment, abs(l): l > 0}
        )

    # Branch
    dpll.decisions += 1
    l = clauses[0][0]
    var = abs(l)
    for polarity in [1, -1]:
        lit = var * polarity
        result = dpll(
            simplify(clauses, lit),
            {**assignment, var: polarity > 0}
        )
        if result is not None:
            return result

    return None



def reset_counters():
    dpll.calls = 0
    dpll.pures = 0
    dpll.propagations = 0
    dpll.decisions = 0


def satisfiable(clauses):
    return dpll(clauses) is not None
