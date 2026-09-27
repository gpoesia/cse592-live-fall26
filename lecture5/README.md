# Lecture 5: DPLL for SAT solving

This directory contains the self-contained code from the Lecture 5 live-coding
session.

We implemented DPLL and used it to solve the [Lights Out](https://en.wikipedia.org/wiki/Lights_Out_\(game\)) puzzle.

## Setup

Python 3.10 or later is recommended. The solver and the demo use only the standard library.

```bash
python lights_out.py
```

Each run shows a random solvable board and a random unsolvable board.

For generating the video that illustrates the solution actually being applied step-by-step,
we also use `pygame-ce` (Python library) and `ffmpeg` (program):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python visualize.py -o out
```

Installing [ffmpeg](https://ffmpeg.org/download.html) is optional (only needed to generate the video).

## What we did in class

1. **Defined a simple representation for CNF formulas.** A formula is a list of clauses, and a clause is a
   list of literals, each one represented as an int, with the sign representing whether its negated or not.
   Example: `7` means "variable 7 is true" and `-7` means "variable 7 is false", [3, -7, 8] represents
    the clause  "variable 3 OR (NOT variable 7) OR variable 8".
    So CNF -> list of lists of ints.

2. **Implemented `simplify`.** Assuming a literal is true, we remove all clauses containing that literal (now satisfied),
    and delete its negation from the other clauses (since the negation of that literal is now false and can't be used to satisfy those clauses).
   If a clause becomes empty, we return None (unsat).

3. **Implemented DPLL.** The main backtracking search:
   - **Base cases.** Empty clause: contradiction (unsat), empty list of clauses: we're done (sat)
   - **Unit propagation.**
   - **Pure literal elimination.**
   - **Split.** Guess a literal, and if we fail also try its negation.

4. **Solved Lights Out.** `python lights_out.py`
    We talked through the encoding of the problem as a SAT problem,
    which is described in `lights_out.py`.
    The variables represent the decision of pressing each cell (or not),
    and the clauses rule out the choices that leave each light on at the end.
