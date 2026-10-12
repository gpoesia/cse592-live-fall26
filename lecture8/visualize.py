"""Generate the gradient descent videos:

    python visualize.py -o out

Needs numpy, pygame-ce and ffmpeg. Each video has three panels that advance one
step per frame: the data with the current lines, the training loss landscape
over w = [w1, w2] with each run's trajectory, and the loss curves.
A GD step uses the whole dataset; an SGD step uses a single example.

    out/gd_vs_sgd.mp4    gradient descent vs SGD
    out/step_sizes.mp4   gradient descent with a small, good, and too-large step size
"""

import argparse
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pygame

from gradient_descent import generate_data, phi, train_gd, train_loss, train_sgd

BACKGROUND = (255, 255, 255)
TEXT = (30, 32, 40)
GRID = (205, 208, 215)
POINT = (90, 94, 105)
INK = (70, 90, 170)
BLUE = (50, 110, 220)
ORANGE = (235, 140, 40)
GREEN = (60, 170, 90)
RED = (215, 70, 70)

PANEL = 380
GAP = 36
HEADER = 80
FOOTER = 40


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


class Panel:
    """Maps data coordinates to pixels inside a square panel."""

    def __init__(self, left, top, xlim, ylim):
        self.rect = pygame.Rect(left, top, PANEL, PANEL)
        self.xlim, self.ylim = xlim, ylim

    def to_px(self, x, y):
        u = (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
        v = (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0])
        # Clamp far-away points (e.g., a diverging run) to keep pygame's ints valid.
        px = min(max(self.rect.left + u * PANEL, -10000), 10000)
        py = min(max(self.rect.top + (1 - v) * PANEL, -10000), 10000)
        return px, py

    def contains(self, x, y):
        return (self.xlim[0] <= x <= self.xlim[1]
                and self.ylim[0] <= y <= self.ylim[1])


class RegressionVisualizer(Animation):
    """Left: data and lines. Middle: loss landscape. Right: loss curves.

    runs: list of (name, color, ws, losses, thin), with ws and losses as returned
    by train_gd() or train_sgd(). Thin runs (many small steps) are drawn below.

    With synchronize=True, runs with fewer steps advance more slowly, so that
    all runs finish together (e.g., one GD step per pass of SGD over the data).
    """

    def __init__(self, x, y, runs, caption, seconds=20, synchronize=False):
        self.steps = max(len(ws) - 1 for _, _, ws, _, _ in runs)
        fps = max(6, round(self.steps / seconds))
        width = 3 * PANEL + 4 * GAP
        super().__init__(width, HEADER + PANEL + FOOTER, fps, "gradient descent")
        self.x, self.y = x, y
        self.runs = runs
        self.draw_order = sorted(runs, key=lambda run: not run[4])
        self.xlabel = "examples processed" if synchronize else "step"
        # How many frames each step of a run lasts.
        self.stride = {name: self.steps / (len(ws) - 1) if synchronize else 1
                       for name, _, ws, _, _ in runs}
        self.caption = caption
        self.num_frames = self.steps + 1 + 2 * fps
        self.best_w = np.linalg.lstsq(phi(x), y)[0]

        # A square window over w containing the start [0, 0] and the minimum.
        center = self.best_w / 2
        half = np.abs(self.best_w).max() / 2 + 1.0
        self.w1_range = (center[0] - half, center[0] + half)
        self.w2_range = (center[1] - half, center[1] + half)

        margin = 0.15 * (y.max() - y.min())
        self.data_panel = Panel(GAP, HEADER, (0, 4),
                                (y.min() - margin, y.max() + margin))
        self.loss_panel = Panel(2 * GAP + PANEL, HEADER, self.w1_range, self.w2_range)
        # The loss curves show how far each run is from the best possible loss.
        self.best_loss = train_loss(self.best_w, x, y)
        gaps = [l - self.best_loss for _, _, _, losses, _ in runs for l in losses]
        low = max(math.floor(math.log10(max(min(gaps), 1e-12))), -6)
        high = math.ceil(math.log10(gaps[0])) + 1
        self.curve_panel = Panel(3 * GAP + 2 * PANEL, HEADER, (0, self.steps),
                                 (low, high))
        self.background = None

    def render_landscape(self):
        """Loss at every pixel, drawn as filled contours: darker = higher loss."""
        w1 = np.linspace(*self.w1_range, PANEL)
        w2 = np.linspace(*self.w2_range, PANEL)[::-1]
        W1, W2 = np.meshgrid(w1, w2, indexing="xy")
        predictions = W1[..., None] + W2[..., None] * self.x
        loss = np.mean((predictions - self.y) ** 2, axis=-1)
        level = np.log(loss / loss.min()) / np.log(loss.max() / loss.min())
        level = np.floor(level * 14) / 14
        white, ink = np.array(BACKGROUND), np.array(INK)
        rgb = white + (ink - white) * level[..., None]
        # surfarray indexes pixels as [x][y], so we transpose rows and columns.
        return pygame.surfarray.make_surface(rgb.transpose(1, 0, 2).astype(np.uint8))

    def draw_axes(self, surface, panel, xlabel, ylabel, title):
        pygame.draw.rect(surface, TEXT, panel.rect, 1)
        text = self.font.render(title, True, TEXT)
        surface.blit(text, text.get_rect(midbottom=(panel.rect.centerx, panel.rect.top - 6)))
        text = self.small.render(xlabel, True, TEXT)
        surface.blit(text, text.get_rect(midtop=(panel.rect.centerx, panel.rect.bottom + 6)))
        text = pygame.transform.rotate(self.small.render(ylabel, True, TEXT), 90)
        surface.blit(text, text.get_rect(midright=(panel.rect.left - 6, panel.rect.centery)))

    def index(self, name, history, t):
        """Which entry of a run's history to show at frame t."""
        return min(int(t / self.stride[name] + 1e-9), len(history) - 1)

    def draw_line(self, surface, w, color, width=3):
        p = self.data_panel
        start = p.to_px(p.xlim[0], w[0] + w[1] * p.xlim[0])
        end = p.to_px(p.xlim[1], w[0] + w[1] * p.xlim[1])
        pygame.draw.line(surface, color, start, end, width)

    def draw(self, surface, frame):
        if self.background is None:
            self.background = self.render_landscape()
            self.font = pygame.font.Font(None, 28)
            self.small = pygame.font.Font(None, 24)
        t = min(frame, self.steps)
        surface.fill(BACKGROUND)

        # Data and the current line of each run.
        p = self.data_panel
        surface.set_clip(p.rect)
        for xi, yi in zip(self.x, self.y):
            pygame.draw.circle(surface, POINT, p.to_px(xi, yi), 5)
        for name, color, ws, _, _ in self.runs:
            self.draw_line(surface, ws[self.index(name, ws, t)], color)
        surface.set_clip(None)
        self.draw_axes(surface, p, "x", "y", "data")

        # Loss landscape with each trajectory, cut off where it leaves the panel.
        p = self.loss_panel
        surface.blit(self.background, p.rect)
        pygame.draw.circle(surface, TEXT, p.to_px(*self.best_w), 6, 2)
        # Trajectories, clipped to the panel; the marker shows the current w.
        surface.set_clip(p.rect)
        for name, color, ws, _, thin in self.draw_order:
            i = self.index(name, ws, t)
            current = ws[i]
            points = [p.to_px(*w) for w in ws[:i + 1]]
            if len(points) > 1:
                pygame.draw.lines(surface, color, False, points, 1 if thin else 2)
            if not thin:
                for point in points:
                    pygame.draw.circle(surface, color, point, 3)
            pygame.draw.circle(surface, color, p.to_px(*current), 7)
        surface.set_clip(None)
        self.draw_axes(surface, p, "w1 (intercept)", "w2 (slope)", "TrainLoss(w)")

        # Loss curves, log scale.
        p = self.curve_panel
        for exponent in range(math.ceil(p.ylim[0]), math.floor(p.ylim[1]) + 1):
            _, py = p.to_px(0, exponent)
            pygame.draw.line(surface, GRID, (p.rect.left, py), (p.rect.right, py))
            label = self.small.render(f"1e{exponent}", True, POINT)
            surface.blit(label, label.get_rect(topright=(p.rect.right - 4, py + 2)))
        surface.set_clip(p.rect)
        for name, color, _, losses, thin in self.draw_order:
            stride = self.stride[name]
            curve = [p.to_px(i * stride, min(math.log10(max(l - self.best_loss, 1e-12)), p.ylim[1] + 1))
                     for i, l in enumerate(losses[:self.index(name, losses, t) + 1])]
            if len(curve) > 1:
                pygame.draw.lines(surface, color, False, curve, 1 if thin else 3)
        surface.set_clip(None)
        self.draw_axes(surface, p, self.xlabel, "TrainLoss - least squares loss (log)", "loss curves")

        # Legend on a white backing (drawn last, so curves never cover it).
        legend = pygame.Rect(p.rect.left + 1, p.rect.top + 1, 200, 24 * len(self.runs) + 8)
        legend.bottom = p.rect.bottom - 1
        pygame.draw.rect(surface, BACKGROUND, legend)
        for row, (name, color, _, losses, _) in enumerate(self.runs):
            loss = losses[self.index(name, losses, t)]
            top = legend.top + 6 + 24 * row
            pygame.draw.rect(surface, color, (legend.left + 8, top + 2, 14, 14))
            label = self.small.render(f"{name}: {loss:.3g}", True, TEXT)
            surface.blit(label, (legend.left + 28, top))

        surface.blit(self.font.render(f"{self.caption}    step {t}", True, TEXT), (GAP, 10))
        pygame.draw.circle(surface, TEXT, (self.width - GAP - 150, 18), 6, 2)
        surface.blit(self.small.render("= least squares", True, TEXT),
                     (self.width - GAP - 138, 10))


def run(name, color, x, y, train, eta, T):
    print(f"\n{name}")
    ws, losses = train(x, y, eta=eta, T=T)
    return name, color, ws, losses, train is train_sgd


def generate(output_dir, T=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y, _ = generate_data()

    runs = [
        run("GD, eta = 0.1", BLUE, x, y, train_gd, 0.1, T),
        run("SGD, eta = 0.01", ORANGE, x, y, train_sgd, 0.01, T),
    ]
    RegressionVisualizer(x, y, runs, "GD vs SGD (one GD step per pass of SGD over the data)",
                         synchronize=True).save(output_dir / "gd_vs_sgd.mp4")

    runs = [
        run("eta = 0.001", GREEN, x, y, train_sgd, 0.001, T),
        run("eta = 0.03", ORANGE, x, y, train_sgd, 0.03, T),
#        run("eta = 0.08", BLUE, x, y, train_sgd, 0.08, T),
        run("eta = 0.15", BLUE, x, y, train_sgd, 0.15, T),
    ]
    RegressionVisualizer(x, y, runs, "SGD: step size"
                         ).save(output_dir / "step_sizes.mp4")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--output-dir", default="out")
    parser.add_argument("-T", type=int, default=10, help="number of passes over the data")
    args = parser.parse_args()
    generate(args.output_dir, T=args.T)


if __name__ == "__main__":
    main()
