# J - JIT Compilation
# A - Automatic Differentiation
# X - XLA (Accelerated Linear Algebra)

# JAX as NumPy
import jax
import jax.numpy as jnp

a = jnp.array([1, 2, 3])
b = jnp.array([4, 5, 6])

print(a + b)
print(jnp.sqrt(a))
print(jnp.mean(a))
print(b.reshape(-1, 1))

# JIT Compilation

# Automatic Differentiation

# Automatic Vectorization

# Randomness
