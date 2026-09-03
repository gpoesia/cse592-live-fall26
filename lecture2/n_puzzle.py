from environment import Environment

# Moves named from the point of view of the piece that will occupy the blank.
MOVES = {
    "down": (-1, 0),
    "up": (1, 0),
    "right": (0, -1),
    "left": (0, 1),
}


class NPuzzle(Environment):
    """
    Classical Sliding-tile puzzle (N-Puzzle).

    A state is a tuple of S^2 integers (here, the 15-puzzle would be size=4, 8-puzzle size=3).

    0 represents the blank.

    Thus, for the 8-puzzle (size=3), the goal state is:

    >>> (1, 2, 3,
         4, 5, 6,
         7, 8, 0)
    """

    def __init__(self, start, size=3, log=False):
        """
            start: starting state, as a tuple as described above.
            size: length of the side of the puzzle's grid (e.g., 8-puzzle -> 3x3 square, so size=3).
        """
        super().__init__(log)
        expected = set(range(size * size))
        if len(start) != size * size or set(start) != expected:
            raise ValueError(f"`start` must contain each number from 0 to {size * size - 1}")
        self.size = size
        self.start = tuple(start)
        self.goal = tuple(list(range(1, size * size)) + [0])

    def initial_state(self):
        return self.start

    def actions(self, state):
        blank = state.index(0)
        row, col = divmod(blank, self.size)
        result = []
        for name, (dr, dc) in MOVES.items():
            if 0 <= row + dr < self.size and 0 <= col + dc < self.size:
                result.append((name, 1))
        return result

    def _transition(self, state, action):
        blank = state.index(0)
        row, col = divmod(blank, self.size)
        dr, dc = MOVES[action]
        swap = (row + dr) * self.size + (col + dc)
        tiles = list(state)
        tiles[blank], tiles[swap] = tiles[swap], tiles[blank]
        next_state = tuple(tiles)
        reward = 1 if self.is_goal(next_state) else 0
        return next_state, reward

    def is_goal(self, state):
        return state == self.goal

    def heuristic(self, state):
        return manhattan(self, state)


def misplaced(env, state):
    """Number of nonblank tiles that are not in their goal position."""
    return sum(
        1
        for index, tile in enumerate(state)
        if tile != 0 and tile != env.goal[index]
    )


def manhattan(env, state):
    """Sum of each tile's row and column distance from its goal."""
    total = 0
    for index, tile in enumerate(state):
        if tile == 0:
            continue
        row, col = divmod(index, env.size)
        goal_row, goal_col = divmod(tile - 1, env.size)
        total += abs(row - goal_row) + abs(col - goal_col)
    return total
