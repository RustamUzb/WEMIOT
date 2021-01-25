import jax.numpy as jnp
from jax import jacfwd, jacrev, grad

class MatReorderLevel:

  def __init__(self, *args, **kwargs):
      self.cq = 5.0
      self.ch = 0.25
      self.d = 1000

  def mat(self, q):
    return ((self.d * self.cq) / q) + ((q * self.ch) / 2)

  def getReorderLevel(self, q):
    gradL = grad(self.mat)
    gradL2 = grad(self.mat)

    epoch = 0
    total = 1
    while not (total < 0.01 or epoch > 200):
      q =  q - gradL(q) / gradL2(q)
      epoch += 1
      total = gradL(q)
      print(q, total)