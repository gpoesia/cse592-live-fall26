"""Solving Lights Out using DPLL.

We play in a grid of cells containing lights, where the goal is to turn all the lights off.
Pressing on a cell (a light) flips it and its neighbors (up, down, left, right): on -> off and off -> on.

Pressing a light twice cancels out, and the order doesn't matter,
so we only need to decide which cells to press.

We thus use one Boolean variable per cell to represent the decision: do we press it or not?
"""
import argparse
import random
from itertools import product

from dpll import dpll, reset_counters

SIZE = 5


def press_variable(row, col):
    """Give each cell one SAT variable from 1 to SIZE * SIZE."""
    # On a 5x5 board, the first row uses variables 1..5, the next 6..10, etc.
    # Variable 1 being True means we decide to press (0, 0), not that its light is initially on.
    return 1 + row * SIZE + col


def cross(row, col):
    """Returns a list with the cell itself and all of its neighbors that are inside the board."""
    candidates = [
        (row, col),
        (row - 1, col),  # up
        (row + 1, col),  # down
        (row, col - 1),  # left
        (row, col + 1),  # right
    ]
    cells = []
    for r, c in candidates:
        if 0 <= r < SIZE and 0 <= c < SIZE:
            cells.append((r, c))
    return cells


def parity_clauses(variables, light_is_on):
    """Returns a clause that forces the press/not press choices to turn one light off.

    An on light needs an odd number of flips (1, 3, 5).
    An off light needs an even number of flips (0, 2, 4).
    """
    clauses = []

    # List all True/False combinations for the presses affecting this light.
    # There are at most five of them, so at most 2**5 combinations to check.
    for values in product([False, True], repeat=len(variables)):
        number_of_presses = sum(values)
        odd_number_of_presses = number_of_presses % 2 == 1
        if odd_number_of_presses == light_is_on:
            continue  # This combination turns the light off, allow it.

        # This combination leaves the light on, so we add a clause to disallow it.
        # A clause is an OR, so it fails only when ALL its literals are false.
        # For example, to forbid (a=True, b=False, c=True), we add:
        #     (not a OR b OR not c), represented as [-a, b, -c].
        # Any other assignment makes at least one of these literals true.
        clause = []
        for variable, is_pressed in zip(variables, values):
            if is_pressed:
                clause.append(-variable)
            else:
                clause.append(variable)
        clauses.append(clause)

    return clauses


def board_to_cnf(board):
    """Encode the board as CNF. A 1 means on, a 0 means off."""
    clauses = []
    for row in range(SIZE):
        for col in range(SIZE):
            # Only presses at this cell and its neighbors can flip this light.
            # E.g., light (0, 0) is affected by presses at (0, 0), (1, 0), (0, 1).
            variables = []
            for press_row, press_col in cross(row, col):
                variables.append(press_variable(press_row, press_col))

            light_is_on = board[row][col] == 1
            clauses.extend(parity_clauses(variables, light_is_on))

    # A CNF formula is an AND of clauses. Satisfying all these clauses means
    # that every light ends up off, using the same set of presses.
    return clauses


def solve(board):
    """Returns the cells to press or None if it's impossible to win."""
    clauses = board_to_cnf(board)
    assignment = dpll(clauses, {})
    if assignment is None:
        return None

    # Translate the True variables back into board coordinates.
    presses = []
    for row in range(SIZE):
        for col in range(SIZE):
            variable = press_variable(row, col)
            # If DPLL leaves a variable unassigned, its value doesn't matter (we use False).
            if assignment.get(variable, False):
                presses.append((row, col))
    return presses


def toggle(board, cell):
    """Return a new board with this cell and its neighbors flipped."""
    new_board = [row.copy() for row in board]
    row, col = cell
    for r, c in cross(row, col):
        new_board[r][c] = 1 - new_board[r][c]
    return new_board


def random_board(rng, presses=8):
    """Make a solvable board by pressing random cells on an empty board."""
    board = [[0] * SIZE for _ in range(SIZE)]
    for _ in range(presses):
        row = rng.randrange(SIZE)
        col = rng.randrange(SIZE)
        board = toggle(board, (row, col))
    return board


def random_configuration(rng):
    """Choose on or off independently for each light."""
    board = []
    for row in range(SIZE):
        lights = []
        for col in range(SIZE):
            lights.append(rng.randint(0, 1))
        board.append(lights)
    return board


def print_board(board, pressing=None):
    """Print the lights, optionally marking the next press with X."""
    for row in range(SIZE):
        symbols = []
        for col in range(SIZE):
            if (row, col) == pressing:
                symbols.append("X")
            elif board[row][col] == 1:
                symbols.append("#")
            else:
                symbols.append(".")
        print("    " + " ".join(symbols))


def print_stats(board):
    n = SIZE * SIZE
    print(f"\n    {n} variables, {len(board_to_cnf(board))} clauses")
    print(f"    2**{n} = {2**n:,} possible assignments")
    print(f"    DPLL: {dpll.calls} calls, {dpll.propagations} propagations, "
          f"{dpll.pures} pure literals, {dpll.decisions} decisions\n")


def demo_solvable(rng):
    print("Solvable board (made by pressing 8 random cells):\n")
    board = random_board(rng)
    print_board(board)

    reset_counters()
    presses = solve(board)
    print_stats(board)
    print(f"    SAT: found a solution with {len(presses)} presses.\n")

    for step, cell in enumerate(presses, 1):
        print(f"    Press {step}: {cell}")
        print_board(board, pressing=cell)
        print()
        board = toggle(board, cell)
        print("    After pressing:")
        print_board(board)
        print()
    print("    All lights off.\n")


def demo_unsolvable(rng):
    print("Unsolvable board:\n")
    while True:
        board = random_configuration(rng)
        reset_counters()
        if solve(board) is None:
            break
    print_board(board)
    print_stats(board)
    print("    UNSAT: there is no set of presses that turns all the lights off.\n")


def main():
    parser = argparse.ArgumentParser(description="Lights Out with DPLL")
    parser.add_argument("--seed", type=int, help="repeat a run with the same boards")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print("Lights Out: # = on, . = off, X = press here")
    print("Coordinates are (row, column), starting at 0.\n")
    demo_solvable(rng)
    demo_unsolvable(rng)


if __name__ == "__main__":
    main()
