from __future__ import annotations

from functools import reduce
from typing import Union, SupportsFloat
from inspect import currentframe


class Var:
    def __init__(self, val: SupportsFloat, grad_fn: callable[[], List[tuple[Var, float]]] = lambda: []):
        self.v, self.grad_fn = float(val), grad_fn
        self.grads = self.grad()

    def __add__(self, val: Union[Var, SupportsFloat]) -> Var:
        other = Var(val) if isinstance(val, SupportsFloat) else val
        return Var(val=self.v + other.v, grad_fn=lambda: [(self, 1.0), (other, 1.0)])

    def __mul__(self, val: Union[Var, SupportsFloat]) -> Var:
        other = Var(val) if isinstance(val, SupportsFloat) else val
        return Var(val=self.v * other.v, grad_fn=lambda: [(self, other.v), (other, self.v)])

    def grad(self, *vars: Var, bp: float = 1.0, **vals: SupportsFloat) -> dict[Var, float]:
        if vals: f = currentframe().f_back; vars += tuple(map((f.f_globals | f.f_locals).__getitem__, vals))
        return reduce(lambda a, b: {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)},
                      ({k: g * val * bp for k, g in input.grads.items()} for input, val in self.grad_fn()),
                      {self: bp}) if not vars else {v: self.grads[v] for v in vars}

x = Var(1)
y = Var(2)
f = (x * x * 3 + y * y * y) * 2
grads = f.grad(x=1, y=2)

print(f.v)       # 22.0
print(grads[x])  # 12.0
print(grads[y])  # 24.0
