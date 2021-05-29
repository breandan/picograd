from functools import reduce


class Var:
  def __init__(self, val, grad_fn=lambda: []):
    self.v, self.grad_fn = val, grad_fn
    self.grads = self.grad()

  def __add__(self, other):
    return Var(val=self.v + other.v, grad_fn=lambda: [(self, 1.0), (other, 1.0)])

  def __mul__(self, other):
    return Var(val=self.v * other.v, grad_fn=lambda: [(self, other.v), (other, self.v)])

  def grad(self, *vars, bp=1.0):  # TODO: incrementalize?
    return reduce(lambda a, b: {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)},
                  [input.grad(*vars, bp=val * bp) for (input, val) in self.grad_fn()],
                  {self: bp}) if not vars else {v: self.grads[v] for v in vars}


x = Var(1.)
y = Var(2.)
f = x * x + y * y * y
print(f.v)  # 9.0
grads = f.grad(x, y)  # TODO: possible to support f.grad(x=1., y=2.)?
print(grads[x])  # 2.0
print(grads[y])  # 12.0
