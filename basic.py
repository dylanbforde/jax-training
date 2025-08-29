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

# cannot just assign positions
# a[1] = 10
# Jax arrays are immutable and do not support in-place assignment
# instead do this
a = a.at[1].set(10)

# JIT Compilation
import time

@jax.jit
def myfunction(x):
    return jnp.where(x % 2 == 0, x / 2, 3 * x + 1) # Collatz Conjecture

arr = jnp.arange(10)

_ = myfunction(arr) # warm up for jit

# this blocks the end line from running until the value is computed
start = time.perf_counter()
myfunction(arr).block_until_ready()
end = time.perf_counter()
print(f'block until ready: {end-start}')

# this does not block the end line from running until the value is computed
# essentially the next python line is ran before the value is computed
start = time.perf_counter()
myfunction(arr)
end = time.perf_counter()
print(f'not blocked until ready: {end-start}')


# Automatic Differentiation

# Automatic Vectorization

# Randomness
