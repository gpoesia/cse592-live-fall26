"""
Generate the n-puzzle solution and search-tree videos used in Lecture 2.

This file was generated using Claude Code (Fable 5).
"""

import argparse
import math
import os
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import pygame

from a_star import AStar
from n_puzzle import NPuzzle, manhattan, misplaced


WHITE = (245, 245, 245)
BLACK = (30, 30, 30)
GRAY = (170, 170, 170)
LIGHT = (210, 215, 225)
BLUE = (60, 120, 230)
GREEN = (70, 180, 95)


class Animation:
    num_frames = 0

    def __init__(self, width, height, fps, title):
        self.width = width - width % 2
        self.height = height - height % 2
        self.fps = fps
        self.title = title

    def draw(self, surface, frame):
        raise NotImplementedError

    def save(self, path):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to generate videos")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        pygame.init()
        surface = pygame.Surface((self.width, self.height))
        with tempfile.TemporaryDirectory() as folder:
            for frame in range(self.num_frames):
                self.draw(surface, frame)
                pygame.image.save(surface, os.path.join(folder, f"{frame:05d}.png"))
            pygame.quit()
            subprocess.run(
                [
                    "ffmpeg", "-y", "-framerate", str(self.fps),
                    "-i", os.path.join(folder, "%05d.png"),
                    "-pix_fmt", "yuv420p", str(path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        print(f"saved {path}")


class NPuzzleVisualizer(Animation):
    def __init__(self, env, actions, cell=110, fps=2):
        side = env.size * cell
        super().__init__(side, side, fps, "n-puzzle solution")
        self.env = env
        self.cell = cell
        self.states = self._rollout(actions)
        self.num_frames = len(self.states)
        self.font = None

    def _rollout(self, actions):
        state = self.env.initial_state()
        states = [state]
        for action in actions:
            state, _ = self.env.transition(state, action)
            states.append(state)
        return states

    def draw(self, surface, frame):
        if self.font is None:
            self.font = pygame.font.SysFont(None, self.cell // 2)
        surface.fill(BLACK)
        state = self.states[frame]
        solved = self.env.is_goal(state)
        for index, tile in enumerate(state):
            if tile == 0:
                continue
            row, col = divmod(index, self.env.size)
            rect = pygame.Rect(
                col * self.cell + 3,
                row * self.cell + 3,
                self.cell - 6,
                self.cell - 6,
            )
            pygame.draw.rect(surface, GREEN if solved else LIGHT, rect, border_radius=8)
            label = self.font.render(str(tile), True, BLACK)
            surface.blit(label, label.get_rect(center=rect.center))


class SearchTreeVisualizer(Animation):
    def __init__(self, log, target_seconds=12, fps=20, width=1000, height=700):
        if not log:
            raise ValueError("search log is empty")
        super().__init__(width, height, fps, "search tree")
        self._build(log)
        frames = max(1, int(target_seconds * fps))
        self.batch = max(1, math.ceil(len(self.edges) / frames))
        self.num_frames = math.ceil(len(self.edges) / self.batch) + 1
        self.radius = max(1, round(6 - math.log10(max(len(self.edges), 1))))
        self.font = self.canvas = None
        self.drawn = 0

    def _build(self, log):
        children = defaultdict(list)
        depth, parent = {}, {}
        self.edges = []
        self.goal = self.goal_index = None
        self.root = log[0].state
        depth[self.root] = 0
        seen = {self.root}

        for transition in log:
            child = transition.next_state
            if child not in seen:
                seen.add(child)
                depth[child] = depth[transition.state] + 1
                parent[child] = transition.state
                children[transition.state].append(child)
                self.edges.append((transition.state, child))
            if transition.reward > 0 and self.goal is None:
                self.goal = child
                self.goal_index = len(self.edges) - 1

        x, next_leaf = {}, 0
        stack = [(self.root, False)]
        while stack:
            node, visited = stack.pop()
            if visited:
                kids = children[node]
                if kids:
                    x[node] = (x[kids[0]] + x[kids[-1]]) / 2
                else:
                    x[node] = next_leaf
                    next_leaf += 1
            else:
                stack.append((node, True))
                for child in reversed(children[node]):
                    stack.append((child, False))

        margin = 30
        goal_depth = depth.get(self.goal)
        if goal_depth:
            spine = {}
            node = self.goal
            while node is not None:
                spine[depth[node]] = x[node]
                node = parent.get(node)
            centered = {
                state: x[state] - spine.get(depth[state], spine[goal_depth])
                for state in x
            }
            row_height = (self.height * 0.85 - margin) / goal_depth
        else:
            midpoint = (min(x.values()) + max(x.values())) / 2
            centered = {state: x[state] - midpoint for state in x}
            max_depth = max(depth.values()) or 1
            row_height = (self.height - 2 * margin) / max_depth

        half_width = max((abs(value) for value in centered.values()), default=1) or 1
        self.positions = {
            state: (
                int(self.width / 2 + (self.width / 2 - margin) * centered[state] / half_width),
                int(margin + depth[state] * row_height),
            )
            for state in x
        }

    def _paint(self, upto):
        for parent, child in self.edges[self.drawn:upto]:
            pygame.draw.line(self.canvas, LIGHT, self.positions[parent], self.positions[child], 1)
            pygame.draw.circle(self.canvas, GRAY, self.positions[child], self.radius)
        self.drawn = upto

    def draw(self, surface, frame):
        if self.canvas is None:
            self.canvas = pygame.Surface((self.width, self.height))
            self.canvas.fill(WHITE)
            pygame.draw.circle(self.canvas, GRAY, self.positions[self.root], self.radius)
            self.font = pygame.font.SysFont(None, 26)

        upto = min(len(self.edges), frame * self.batch)
        self._paint(upto)
        surface.blit(self.canvas, (0, 0))

        for _parent, child in self.edges[max(0, upto - self.batch):upto]:
            pygame.draw.circle(surface, BLUE, self.positions[child], self.radius + 1)
        if self.goal_index is not None and upto > self.goal_index:
            pygame.draw.circle(surface, GREEN, self.positions[self.goal], self.radius + 3)

        label = self.font.render(f"states discovered: {upto:,}", True, BLACK)
        surface.blit(label, (10, 10))


def scramble(moves, seed, size):
    random_generator = random.Random(seed)
    env = NPuzzle(list(range(1, size * size)) + [0], size=size)
    state = env.initial_state()
    for _ in range(moves):
        action, _ = random_generator.choice(env.actions(state))
        state, _ = env.transition(state, action)
    return list(state)


def generate(output_dir, quick=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if quick:
        start, size, seconds = [1, 2, 3, 4, 0, 6, 7, 5, 8], 3, 1
    else:
        start, size, seconds = scramble(30, seed=1, size=4), 4, 12

    solution_env = NPuzzle(start, size=size)
    path = AStar(lambda _env, _state: 0).search(solution_env)
    NPuzzleVisualizer(solution_env, path).save(output_dir / "puzzle_solution.mp4")

    algorithms = [
        ("tree_h0", AStar(lambda _env, _state: 0)),
        ("tree_misplaced", AStar(misplaced)),
        ("tree_manhattan", AStar(manhattan)),
    ]
    for name, algorithm in algorithms:
        env = NPuzzle(start, size=size, log=True)
        result = algorithm.search(env)
        print(f"{name:16} length={len(result):2} expanded={algorithm.expanded:,}")
        SearchTreeVisualizer(env.log, target_seconds=seconds).save(output_dir / f"{name}.mp4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", default="out")
    parser.add_argument("--quick", action="store_true", help="render a tiny smoke-test puzzle")
    args = parser.parse_args()
    generate(args.output_dir, quick=args.quick)


if __name__ == "__main__":
    main()
