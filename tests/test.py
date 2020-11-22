import util.utilities as util
import jax.numpy as jnp

print((jnp.log(1 / (1 - util.median_rank(10, 1, 0.05))) ** (1 / 3)) * 1000)
print((jnp.log(1 / (1 - util.median_rank(10, 1, 0.95))) ** (1 / 3)) * 1000)
