"""Train both classifiers with SGD and plot their decision boundaries."""
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model import LinearClassifier, NeuralNet


def generate_data(n=200, noise=0.06, seed=592):
    """Samples from two "spirals" with labels -1 and +1."""
    rng = np.random.default_rng(seed)
    y = np.where(np.arange(n) % 2 == 0, -1, 1)
    radius = rng.uniform(0.15, 1.0, n)
    angle = 2 * np.pi * radius + (y == 1) * np.pi
    x = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    x += rng.normal(0, noise, x.shape)
    return x, y


def squared_hinge(score, y):
    margin = (1 - y * score).relu()
    return margin * margin


def predict(model, xs):
    """Return one score per point using the model's forward pass."""
    return [model(x).value for x in xs]


def train_sgd(model, x, y, epochs=10, eta=0.02, seed=592):
    rng = random.Random(seed)
    examples = list(zip(x, y))
    parameters = model.parameters()
    losses = []

    for epoch in range(epochs):
        step_size = eta * (1 - 0.9 * epoch / epochs)
        rng.shuffle(examples)
        for point, label in examples:
            loss = squared_hinge(model(point), int(label))
            for p in parameters:
                p.grad = 0.0
            loss.backward()
            for p in parameters:
                p.value -= step_size * p.grad

        # Measure loss and accuracy with the updated weights.
        scores = predict(model, x)
        total_loss = 0.0
        correct = 0
        for score, label in zip(scores, y):
            total_loss += max(0, 1 - label * score) ** 2
            if (score >= 0) == (label == 1):
                correct += 1
        losses.append(total_loss / len(examples))
        accuracy = correct / len(examples)
        print(f"epoch {epoch + 1:3d}   loss = {losses[-1]:.4f}   "
              f"accuracy = {accuracy:.1%}", flush=True)

    return losses


def plot_results(models, histories, x, y, test_x, test_y, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    extent = 1.25
    ticks = np.linspace(-extent, extent, 200)
    xx, yy = np.meshgrid(ticks, ticks)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    for ax, (name, model) in zip(axes[:2], models.items()):
        scores = np.array(predict(model, grid)).reshape(xx.shape)
        ax.contourf(xx, yy, scores >= 0, levels=[-0.5, 0.5, 1.5],
                    colors=["#c5d9f0", "#f5c6c6"], alpha=0.8)
        if scores.min() < 0 < scores.max():
            ax.contour(xx, yy, scores, levels=[0], colors=["#333333"], linewidths=1.5)
        ax.scatter(x[:, 0], x[:, 1], c=y, cmap="coolwarm", vmin=-1, vmax=1,
                   s=22, edgecolors="white", linewidths=0.4)
        test_scores = predict(model, test_x)
        accuracy = sum((score >= 0) == (label == 1)
                       for score, label in zip(test_scores, test_y)) / len(test_y)
        ax.set_title(f"{name}\nTest accuracy: {accuracy:.1%}")
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), xlabel="$x_1$", ylabel="$x_2$")
        ax.set_aspect("equal")

    for name, losses in histories.items():
        axes[2].plot(range(1, len(losses) + 1), losses, label=name)
    axes[2].set(xlabel="Epoch", ylabel="Mean squared hinge loss", title="Training with SGD")
    axes[2].legend()
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"\nSaved {output}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eta", type=float, default=0.02, help="initial SGD step size")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--seed", type=int, default=592)
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--noise", type=float, default=0.06)
    parser.add_argument("-o", "--output", default="out/decision_boundaries.png")
    parser.add_argument("--show", action="store_true", help="also open the plot window")
    args = parser.parse_args()
    if args.epochs < 1 or args.width < 1 or args.points < 2 or args.eta <= 0 or args.noise < 0:
        parser.error("epochs, width, eta must be positive; points >= 2; noise >= 0")

    x, y = generate_data(args.points, args.noise, args.seed)
    test_x, test_y = generate_data(1000, args.noise, args.seed + 1)
    models = {"Linear classifier": LinearClassifier(args.seed),
              "ReLU neural net": NeuralNet(args.width, args.seed)}
    histories = {}
    for name, model in models.items():
        print(f"\n{name} ({len(model.parameters())} parameters)", flush=True)
        histories[name] = train_sgd(model, x, y, args.epochs, args.eta, args.seed)
        scores = predict(model, test_x)
        accuracy = sum((score >= 0) == (label == 1)
                       for score, label in zip(scores, test_y)) / len(test_y)
        print(f"Test accuracy: {accuracy:.1%}", flush=True)

    fig = plot_results(models, histories, x, y, test_x, test_y, args.output)
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
