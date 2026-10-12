"""
Scalar automatic differentiation engine.

Vectors and matrices are then built out of scalar parameters.
"""


def as_parameter(p: float | 'Parameter') -> 'Parameter':
    'Wraps p as a Parameter if it is a float.'
    if isinstance(p, Parameter):
        return p
    return Parameter(p)


class Parameter:
    """
    A Parameter will be used to represent a float value that
    we will want to compute gradients for automatically.
    """
    def __init__(self, value, parents=()):
        # TODO: initialize value, grad, parents and _backward
        self.value = value
        self.grad = 0
        self.parents = parents
        self._backward = lambda: None  # no-op

    # p1 + 1.5
    def __add__(self, other):
        # TODO:
        # 1- Make sure other is a Parameter
        # 2- Create Parameter to represent self + other
        # 3- Create backward() for this new parameter
        other = as_parameter(other)

        result = Parameter(self.value + other.value,
                           (self, other))

        def backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = backward
        return result


    def __mul__(self, other):
        # TODO:
        # 1- Make sure other is a Parameter
        # 2- Create Parameter to represent self * other
        # 3- Create backward() for this new parameter
        other = as_parameter(other)

        result = Parameter(self.value * other.value, (self, other))

        def backward():
            self.grad += result.grad * other.value
            other.grad += result.grad * self.value

        result._backward = backward
        return result


    def relu(self):
        # TODO:
        # 1- Create Parameter to represent relu(self)
        # 2- Create backward() for this new parameter
        result = Parameter(max(0, self.value), (self,))

        def backward():
            if self.value > 0:
                self.grad += result.grad

        result._backward = backward
        return result

    __radd__ = __add__   # allows us to add float + Parameter
    __rmul__ = __mul__   # allows us to multiply float * Parameter

    # TODO: unary -x (__neg__)
    def __neg__(self):
        return self * -1

    # TODO: x - y (__sub__)
    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def backward(self):
        # TODO
        # 1- Compute topological order
        # 2- Start with self.grad = 1 (gradient of loss wrt itself)
        # 3- Call _backward() on all nodes in reverse topological order
        # loss.backward()
        visited, order = set(), []

        def visit(node):
            if node not in visited:
                visited.add(node)
                for p in node.parents:
                    visit(p)
                order.append(node)

        visit(self)
        self.grad = 1

        for node in reversed(order):
            node._backward()

    def __repr__(self):
        return f"Parameter(value={self.value:.4f}, grad={self.grad:.4f})"


class Vector:
    def __init__(self, values: list):
        self.values = [as_parameter(v) for v in values]

    def __add__(self, other):
        assert len(self.values) == len(other.values), 'Vector sizes don\'t match'
        return Vector([a + b for a, b in zip(self.values, other.values)])

    def __mul__(self, other):
        if isinstance(other, Vector):
            # Dot product if multiplying two vectors
            assert len(self.values) == len(other.values), 'Vector sizes don\'t match'
            return sum(a * b for a, b in zip(self.values, other.values))
        return Vector([a * other for a in self.values])

    __rmul__ = __mul__

    def relu(self):
        return Vector([a.relu() for a in self.values])

    def parameters(self):
        return self.values


class Matrix:
    def __init__(self, rows):
        self.rows = [row if isinstance(row, Vector) else Vector(row)
                     for row in rows]

    def __mul__(self, vector):
        return Vector([row * vector for row in self.rows])

    def parameters(self):
        return [p for row in self.rows for p in row.parameters()]
