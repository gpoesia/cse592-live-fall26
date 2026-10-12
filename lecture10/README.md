# Lecture 10: Optimization and neural networks

We implemented a scalar autograd engine,
then used it to train a linear classifier
and a 3-layer ReLU neural net.
Both models use the same train/test data, loss, and SGD optimizer.

Credit: this is inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

## Setup

Python 3.10 or later is recommended. The code uses `numpy` and `matplotlib` to visualize the decision boundaries (no video this time!).

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
```

## What we did in class

1. **Implemented the Parameter class in parameter.py.** This represents a single scalar that we can backpropagate through.
    We overrode binary operators like addition and multiplication (__add__, __mul__ methods)
    by both computing the result of the operation and a closure that is called to update
    the gradients of the parent nodes of the resulting `Parameter` in the backward pass.

2. **Implemented backward().**
    In the backward pass, called on a Parameter that represents the loss,
    we 

3. **Walked through how to build vectors and matrices out of a single Parameter.** (also in parameter.py)

4. **Walked through a simple ReLU neural network using our trainable Parameter.** (in model.py)

5. **Trained both a logistic regression classifier and our neural net using SGD.** We saw that the neural net was able to fit a highly non-linear decision boundary.
