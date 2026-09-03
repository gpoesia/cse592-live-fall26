# Lecture 2: Search

In class, we implemented A\* search, ran it on the n-puzzle and visualized
how two different classical heuristics impact performance.

## Setup (for running locally)

Python 3.10 or later is recommended. The search part itself uses only the Python standard library.

For video generation, we additionally need both `pygame-ce` (Python library) and `ffmpeg` (program). For example:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you want to generate the videos yourself, you'll need to [install ffmpeg](https://ffmpeg.org/download.html)
for your system. This is optional.

## What we did in class

1. **Looked at abstract interfaces:** 
   - `environment.py` defines a generic environment.
   - `search.py` defines search nodes and a generic search algorithm.
     Much of the data here (e.g., parent, parent actions) is only needed
     to reconstruct the path at the end of search.
   - `n_puzzle.py` is the example of concrete environment we used.
     The same search algorithms will work with other environments,
     as long as they properly implement the interface we defined.

2. **Implemented A*.**
    In `a_star.py`, we maintained:

   - `frontier`: priority queue of nodes ordered by $f(n)=g(n)+h(n)$;
   - `reached`: map from reached states to the cheapest node, by $g(n)$, found for that state;

   Small implementation detail: Python's heapq needs elements to be all comparable,
   and tuples by default compare each fields from left to right.
   We used a simple counter to serve as a tie-breaker in this case,
   since our Node objects are not comparable with '<' (though we could make them so).

3. **Heuristics for the n-puzzle.** Every misplaced tile must move at
   least once, which gives us the "count misplaced tiles heuristic".
   We implemented this heuristic in `demo.py`;
   this was also already implemented in `n_puzzle.py`.

   But each tile must also traverse the row and column distance to its goal position,
   giving us the stronger Manhattan distance heuristic.
   We looked at the implementation in `n_puzzle.py` and tried it out too.

4. **Compared performance.** We ran `python run_example.py`.

   All three methods find the same optimal solution with 20 moves:

   ```text
   heuristic      length     expanded
   ----------------------------------
   h=0                20       55,409
   misplaced          20        3,669
   manhattan          20          283
   ```

   $h=0$ gives uninformed Uniform-Cost Search (or BFS in this case, since all costs are equal).

   Both heuristics reduce the work quite substantially, with Manhattan distance performing significantly better. It is indeed a very effective heuristic for this puzzle.

5. **Videos.** We ran `python visualize.py -o out` to generate
   the solution animation and three search trees:

   - `out/puzzle_solution.mp4`
   - `out/tree_h0.mp4`
   - `out/tree_misplaced.mp4`
   - `out/tree_manhattan.mp4`
