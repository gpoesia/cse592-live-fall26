from a_star import AStar
from n_puzzle import NPuzzle, manhattan


START = [7, 2, 4,
         5, 0, 6,
         8, 3, 1]

def misplaced(env, s):
    goal = env.goal
    return sum(1 for i in range(len(s)) if goal[i] != s[i])


def compare_heuristics():
    heuristics = [
        ("h=0", lambda _env, _state: 0),
        ("misplaced", misplaced),
        ("manhattan", manhattan),
    ]

    print(f"{'Heuristic':12} {'Length':>8} {'Expanded':>10}")
    print("-" * 40)

    for name, heuristic in heuristics:
        env = NPuzzle(START)
        algorithm = AStar(heuristic)
        path = algorithm.search(env)
        print(f"{name:12} {len(path):8} {algorithm.expanded:10,}")


if __name__ == "__main__":
    compare_heuristics()
