import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, grad, lax
from functools import partial
from torchvision.datasets import MNIST

# ----- Params -----
def random_layer_params(in_dim, out_dim, key, scale=1e-2):
    kW, kb = random.split(key)
    W = scale * random.normal(kW, (in_dim, out_dim))
    b = scale * random.normal(kb, (out_dim,))
    return W, b

def init_network_params(key, sizes):
    keys = random.split(key, len(sizes) - 1)
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

# ----- Model / loss / metrics (Batched) -----
def apply(params, X):         # X: [B, D]
    for W, b in params[:-1]:
        X = jax.nn.relu(X @ W + b)
    Wf, bf = params[-1]
    logits = X @ Wf + bf
    return jax.nn.log_softmax(logits)

def loss(params, X, Y):       # Y: one-hot [B, C]
    logp = apply(params, X)
    return -jnp.mean(jnp.sum(logp * Y, axis=-1))

@jit
def accuracy(params, X, Y):
    preds = jnp.argmax(apply(params, X), axis=-1)
    targ  = jnp.argmax(Y, axis=-1)
    return jnp.mean(preds == targ)

# Donate params to avoid copies on device
@partial(jit, donate_argnums=(0,))
def train_epoch(params, Xb, Yb, step_size):
    def step(params, batch):
        x, y = batch
        g = grad(loss)(params, x, y)
        params = jax.tree_util.tree_map(lambda p, dg: p - step_size * dg, params, g)
        return params, None
    params, _ = lax.scan(step, params, (Xb, Yb))
    return params

def one_hot(x, k, dtype=jnp.float32):
    x = np.asarray(x)
    return jnp.array(x[:, None] == np.arange(k), dtype)

def make_epoch_batches(X, Y, batch_size, rng=None):
    N = X.shape[0]
    perm = np.arange(N) if rng is None else rng.permutation(N)
    N_eff = (N // batch_size) * batch_size   # drop last to keep shapes static
    idx = perm[:N_eff]
    Xs = X[idx].reshape(-1, batch_size, X.shape[1]).astype(np.float32)
    Ys = Y[idx].reshape(-1, batch_size, Y.shape[1]).astype(np.float32)
    return jax.device_put(Xs), jax.device_put(Ys)

# ===== Data (PyTorch MNIST -> NumPy) =====
mnist_train = MNIST('/tmp/mnist/', download=True, train=True)
mnist_test  = MNIST('/tmp/mnist/', download=True, train=False)

X_train = mnist_train.data.numpy().reshape(len(mnist_train), -1).astype(np.float32)
y_train = mnist_train.targets.numpy()
X_test  = mnist_test.data.numpy().reshape(len(mnist_test), -1).astype(np.float32)
y_test  = mnist_test.targets.numpy()

n_classes = 10
Y_train = one_hot(y_train, n_classes)
Y_test  = one_hot(y_test,  n_classes)

# ===== Training =====
layer_sizes = [784, 512, 512, 10]
step_size   = 0.01
num_epochs  = 10
batch_size  = 128

key = random.key(0)
params = init_network_params(key, layer_sizes)

# (Optional) warmup compile
Xb_warm, Yb_warm = make_epoch_batches(X_train[:batch_size*2], Y_train[:batch_size*2], batch_size)
params = train_epoch(params, Xb_warm, Yb_warm, step_size)

start = time.time()
for epoch in range(1, num_epochs+1):
    rng = np.random.default_rng(epoch)
    Xb, Yb = make_epoch_batches(X_train, Y_train, batch_size, rng)
    params = train_epoch(params, Xb, Yb, step_size)
    jax.block_until_ready(params)

    tr = accuracy(params, jax.device_put(X_train), jax.device_put(Y_train))
    te = accuracy(params, jax.device_put(X_test),  jax.device_put(Y_test))
    tr, te = float(tr), float(te)
    print(f"epoch {epoch:02d} | train {tr:.4f} | test {te:.4f}")

print(f"Total Epoch time: {time.time() - start}")
