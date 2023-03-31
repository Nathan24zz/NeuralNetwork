class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data)
        return out

    def __radd__(self, other):  # other + self
        # reverse add operation
        # ex: int + Value -> Value + int
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data)
        return out

    def __rmul__(self, other):  # other * self
        return self * other
