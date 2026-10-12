"""Linear regression with gradient descent and SGD.

Predictor: f_w(x) = w . phi(x), with features phi(x) = [1, x].
Loss: squared loss, TrainLoss(w) = 1/n * (w . phi(x) - y)^2.
"""
import argparse

import numpy as np

rng = np.random.default_rng(0)


def generate_data(n=20, noise=2.0, seed=592):
    """Sample random data from a line + Gaussian noise."""
    data_rng = np.random.default_rng(seed)
    true_w = data_rng.uniform(-2, 2, size=2)
    x = data_rng.uniform(0, 4, size=n)
    y = true_w[0] + true_w[1] * x + data_rng.normal(0, noise, size=n)
    return x, y, true_w


def phi(x):
    """Compute features for each input: one row per example with [1, x]."""
    # Here x is a vector of size N (whole dataset)
    one = np.ones_like(x)
    return np.stack([one, x], axis=-1)
    # Output is a (N, 2) matrix


def predict(w: np.ndarray, x):
    # phi(x) . w
    return np.vecdot(phi(x), w)


def train_loss(w, x, y):
    # Squared loss
    residuals = predict(w, x) - y
    return np.mean(residuals**2)


def gradient(w, x, y):
    """Gradient of the average squared loss over the examples (x, y)."""
    residuals = predict(w, x) - y
    return 2 * phi(x).T @ residuals / len(y)


def train_gd(x, y, eta=0.1, T=50):
    """Run T iterations of gradient descent

    Returns the list of weights and the list of training losses.
    """
    w = np.zeros(2)
    ws = [w]
    losses = [train_loss(w, x, y)]
    print(f"t = {0:3d}   w = [{w[0]:7.3f}, {w[1]:7.3f}]   loss = {losses[0]:.4f}")

    for i in range(T):
        w = w - eta * gradient(w, x, y)
        ws.append(w)
        losses.append(train_loss(w, x, y))
        print(f"t = {i+1:3d}   w = [{w[0]:7.3f}, {w[1]:7.3f}]   loss = {losses[-1]:.4f}")


    return ws, losses


def train_sgd(x, y, eta=0.1, T=50):
    """Run T iterations of SGD"""
    w = np.zeros(2)
    ws = [w]
    losses = [train_loss(w, x, y)]
    print(f"t = {0:3d}   w = [{w[0]:7.3f}, {w[1]:7.3f}]   loss = {losses[0]:.4f}")

    for t in range(T):
        order = list(range(len(x)))
        rng.shuffle(order)
        for i in order:
            w = w - eta * gradient(w, x[i:i+1,], y[i:i+1,])
            ws.append(w)
            losses.append(train_loss(w, x, y))
            print(f"t = {t+1:3d}   w = [{w[0]:7.3f}, {w[1]:7.3f}]   loss = {losses[-1]:.4f}")

    return ws, losses

def main():
    parser = argparse.ArgumentParser(description="Linear regression with GD / SGD")
    parser.add_argument("--optimizer", choices=["gd", "sgd"], default="gd")
    parser.add_argument("--eta", type=float, default=0.1, help="step size")
    parser.add_argument("-T", type=int, default=50, help="number of iterations")
    args = parser.parse_args()

    x, y, true_w = generate_data()
    print(f"True line: w = [{true_w[0]:.3f}, {true_w[1]:.3f}]")
    best_w = np.linalg.lstsq(phi(x), y)[0]
    print(f"Least squares: w = [{best_w[0]:.3f}, {best_w[1]:.3f}]   "
          f"TrainLoss = {train_loss(best_w, x, y):.4f}\n")

    if args.optimizer == 'gd':
        train_gd(x, y, eta=args.eta, T=args.T)
    elif args.optimizer == 'sgd':
        train_sgd(x, y, eta=args.eta, T=args.T)


if __name__ == "__main__":
    main()
