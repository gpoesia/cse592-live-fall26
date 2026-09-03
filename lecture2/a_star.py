import heapq
import itertools

from search import Node, SearchAlgorithm


class AStar(SearchAlgorithm):
    """A*: Best-First Search ordered by f(n) = g(n) + h(n)."""

    def __init__(self, heuristic=None):
        super().__init__()
        self.heuristic = heuristic

    def search(self, env) -> list['Action']:
        self.expanded = 0

        node = Node(env.initial_state())

        # a map from states to the best node (in g(n)) with that
        # state
        reached = {node.state: node}

        # priority queue (min-heap) of tuples
        # (cost, tie-breaking, node)
        it = 0 # counter for tie-breaking
        frontier = [(0, it, node)]

        while frontier:
            _, _, node = heapq.heappop(frontier)

            if env.is_goal(node.state):
                return node.path()

            self.expanded += 1

            for a, a_cost in env.actions(node.state):
                next_s, _ = env.transition(node.state, a)

                g_n = node.cost + a_cost

                if next_s not in reached or \
                   g_n < reached[next_s].cost:
                    f_n = g_n + self.heuristic(env, next_s)

                    next_node = Node(
                        next_s,
                        node,
                        a,
                        g_n,
                    )

                    reached[next_s] = next_node

                    it += 1
                    heapq.heappush(frontier, (f_n, it, next_node))

        return None  # TODO
