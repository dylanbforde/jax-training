# J - JIT Compilation
# A - Automatic Differentiation
# X - XLA (Accelerated Linear Algebra)

# JAX as NumPy
import jax
from jax._src.config import _validate_default_device
import jax.numpy as jnp
import time

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

@jax.jit
def myfunction(x):
    return jnp.where(x % 2 == 0, x / 2, 3 * x + 1)  # Collatz Conjecture


arr = jnp.arange(10)

_ = myfunction(arr)  # warm up for jit

# this blocks the end line from running until the value is computed
start = time.perf_counter()
myfunction(arr).block_until_ready()
end = time.perf_counter()
print(f"block until ready: {end - start}")

# this does not block the end line from running until the value is computed
# essentially the next python line is ran before the value is computed
start = time.perf_counter()
myfunction(arr)
end = time.perf_counter()
print(f"not blocked until ready: {end - start}")

# intermediate representation that jax uses
# print(jax.make_jaxpr(myfunction)(arr))

# not possible to do
# @jax.jit
# def f(x):
#     if x % 2 == 0:
#         return 1
#     else:
#         return 0

# f(10)

# Automatic Differentiation
def square(x):
    return x ** 2

# needs to be a float
value = 10.0
print('automatic differentiation square function')
print(square(value))

print(jax.grad(square)(value))
print(jax.grad(jax.grad(square))(value))
print(jax.grad(jax.grad(jax.grad(square)))(value))


def f(x, y, z):
    return x ** 2 + 2 * y ** 2 + 3 * z ** 2

x, y, z = 2.0, 2.0, 2.0
print('automatic differentiation polynomial')
print(f(x, y, z))
print(jax.grad(f, argnums=0)(x, y, z))
print(jax.grad(f, argnums=1)(x, y, z))
print(jax.grad(f, argnums=2)(x, y, z))

def f_arr(arr):
     return arr[0] ** 2 + 2 * arr[1] ** 2 + 3 * arr[2] ** 2

print('automatic differentiation with list rather than separate values')
print(f_arr([x, y, z]))
print(jax.grad(f_arr)([x, y, z]))
 
# Automatic Vectorization

key = jax.random.key(42)

W = jax.random.normal(key, (150, 100)) # 100 values per input sample, 150 neurons in next layer
X = jax.random.normal(key, (10, 100))

def calculate_output(x):
    return jnp.dot(W, x)

# not the most efficient way
def batched_calculation_la(X):
    return jnp.dot(X, W.T)

batched_calculation_vmap = jax.vmap(calculate_output)

start = time.perf_counter()
batched_calculation_la(X)
end = time.perf_counter()
print(f'la batch calculation: {end-start}')

start = time.perf_counter()
batched_calculation_vmap(X)
end = time.perf_counter()
print(f'vmap batch calculation: {end-start}')

print(jnp.allclose(batched_calculation_vmap(X), batched_calculation_la(X), atol=1E-4, rtol=1E-4))

# Randomness

key = jax.random.key(1)
key1, key2 = jax.random.split(key)
key3, key4 = jax.random.split(key1)
keys = jax.random.split(key, 10)
print(keys)



