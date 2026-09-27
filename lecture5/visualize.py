"""Generate the Lights Out solution video:

    python visualize.py -o out

Needs pygame-ce and ffmpeg. The animation shows the board the solver was given,
then each press in turn: the cell being pressed flashes, and its cross toggles.

Generated using Codex (GPT-5.6 Sol)
"""

import argparse
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import pygame

from dpll import reset_counters
from lights_out import SIZE, cross, random_board, solve, toggle

BACKGROUND = (24, 26, 32)
OFF = (52, 56, 68)
ON = (250, 204, 74)
PRESS = (232, 96, 84)
TEXT = (232, 234, 240)


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


class LightsOutVisualizer(Animation):
    """One frame showing the cell about to be pressed, one showing the result."""

    def __init__(self, board, presses, cell=96, margin=28, hold=3, fps=2):
        side = SIZE * cell + 2 * margin
        super().__init__(side, side + 56, fps, "lights out")
        self.cell = cell
        self.margin = margin
        self.boards = [board]
        for pressed_cell in presses:
            self.boards.append(toggle(self.boards[-1], pressed_cell))
        self.presses = presses

        # Each press gets two frames, with the first and last board held longer.
        self.steps = ([(0, None)] * hold
                      + [(step, cell) for step, cell in enumerate(presses)
                         for _ in range(2)]
                      + [(len(presses), None)] * hold)
        for index in range(hold, len(self.steps) - hold, 2):
            self.steps[index + 1] = (self.steps[index][0] + 1, None)
        self.num_frames = len(self.steps)
        self.font = None

    def draw(self, surface, frame):
        if self.font is None:
            self.font = pygame.font.SysFont(None, 34)
        step, pressing = self.steps[frame]
        board = self.boards[step]
        surface.fill(BACKGROUND)

        for r in range(SIZE):
            for c in range(SIZE):
                rect = pygame.Rect(self.margin + c * self.cell + 4,
                                   self.margin + r * self.cell + 4,
                                   self.cell - 8, self.cell - 8)
                color = ON if board[r][c] else OFF
                pygame.draw.rect(surface, color, rect, border_radius=12)
                if pressing is not None and (r, c) in cross(*pressing):
                    pygame.draw.rect(surface, PRESS, rect, width=4,
                                     border_radius=12)
                if (r, c) == pressing:
                    pygame.draw.circle(surface, PRESS, rect.center, 12)

        if pressing is not None:
            label = f"press {step + 1} of {len(self.presses)}: {pressing}"
        elif step == 0:
            label = "the board the solver was given"
        elif step == len(self.presses):
            label = "all lights off"
        else:
            label = f"after press {step} of {len(self.presses)}"
        text = self.font.render(label, True, TEXT)
        surface.blit(text, text.get_rect(
            center=(self.width // 2, self.height - 30)))


def generate(output_dir, seed=0, quick=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    board = random_board(rng, presses=4 if quick else 8)
    reset_counters()
    presses = solve(board)
    print(f"{len(presses)} presses: " + ", ".join(str(p) for p in presses))
    LightsOutVisualizer(board, presses).save(output_dir / "lights_out.mp4")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output-dir", default="out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="render a shorter smoke-test video")
    args = parser.parse_args()
    generate(args.output_dir, seed=args.seed, quick=args.quick)


if __name__ == "__main__":
    main()
