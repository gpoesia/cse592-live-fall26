# Lecture 8: Linear regression and gradient descent

This directory contains the code from the Lecture 8 live-coding session. We
trained a linear regressin model on noisy data by minimizing the training loss (squared loss)
with full-batch gradient descent (GD) and stochastic gradient descent (SGD).

## Setup

Python 3.10 or later is recommended. The code uses `numpy`.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python gradient_descent.py
```

For the videos, we also use `pygame-ce` (Python library) and `ffmpeg` (program):

```bash
python visualize.py -o out
```

This writes `out/gd_vs_sgd.mp4` and `out/step_sizes.mp4`. Each shows the data
with the current lines, the training loss landscape over `w = [w1, w2]` with
each run's trajectory, and the loss curves. Installing
[ffmpeg](https://ffmpeg.org/download.html) is optional (only needed to generate
the videos).

## What we did in class

1. **Wrote the predictor and the loss.** Features `phi(x) = [1, x]`, predictor
   `f_w(x) = w . phi(x)`, and `TrainLoss(w)`, the average squared error.

2. **Computed the gradient.**
   `gradient(w, x, y)` is the average of `2 (w . phi(x) - y) phi(x)` over the
   examples.

3. **Implemented the optimizers.** `train_gd()` runs gradient descent from
   `w = [0, 0]`, taking one step with the full gradient (over entire dataset) per iteration.
   `train_sgd()` instead takes one step per example on each pass over the data.
   Both print their progress and return the list of weights and losses.

4. **Generated synthetic data.** `generate_data()` samples a random line,
   then points around it with Gaussian noise.

5. **Tried a few step sizes.** On this data, gradient descent diverges when
   `eta` is above about 0.2.
