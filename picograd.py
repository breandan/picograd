class Var:
  def __init__(self, val, grad_fn=lambda: []):
    self.v = val
    self.grad_fn = grad_fn
    
  def __add__(self, other):
    return Var(self.v + other.v,
      lambda: [(self, 1.0), (other, 1.0)])
      
  def __mul__(self, other):
    return Var(self.v * other.v,
      lambda: [(self, other.v), (other, self.v)])
      
  def grad(self, bp = 1.0, dict = {}):
    dict[self] = dict.get(self, 0) + bp
    for input, val in self.grad_fn():
        input.grad(val * bp, dict)
    return dict
