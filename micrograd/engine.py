import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc
        self.label = label  # for testing purpose

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def backward(self):
        # topological sort
        result = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                result.append(v)
        build_topo(self)

        # initialize self gradient to 1.0
        self.grad = 1.0
        # do backward from end to start
        for item in reversed(result):
            item._backward()

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):  # other + self
        # reverse add operation
        # ex: int + Value -> Value + int
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += 1 - out.data**2
        out._backward = _backward
        return out
