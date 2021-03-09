from functools import reduce

class Var:
  def __init__(self, val, grad_fn=lambda: []):
    self.v, self.grad_fn = val, grad_fn

  def __add__(self, other):
    return Var(val=self.v + other.v, grad_fn=lambda: [(self, 1.0), (other, 1.0)])

  def __mul__(self, other):
    return Var(val=self.v * other.v, grad_fn=lambda: [(self, other.v), (other, self.v)])

  # TODO: Can we turn this into self.grad = ...?
  def grad(self, bp=1.0):
    grads = [input.grad(bp=val * bp) for (input, val) in self.grad_fn()]
    merge = lambda a, b: {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}
    return reduce(merge, grads, {self: bp})

x = Var(1.)
y = Var(2.)
f = x * x + y * y * y
print(f.v)
print(f.grad()[x]) # TODO: Alternate syntax f.grad(x,y) -> Dict?
print(f.grad()[y])