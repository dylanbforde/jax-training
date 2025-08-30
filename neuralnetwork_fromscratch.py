import jax
import jax.numpy as jnp
import numpy as np
import time

from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map

from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import MNIST

# helper function to initialise w + b
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n,m)), scale * random.normal(b_key, (n, ))

# initialise all layers for a fully connected neural network with sizes "sizes"
def init_network_params(key, sizes):
    keys = random.split(key, len(sizes)-1)
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

layer_sizes = [784, 512, 512, 10]
step_size = 0.01
num_epochs = 10
batch_size = 128
n_targets = 10
params = init_network_params(random.key(0), layer_sizes)

def relu(x):
    return jnp.maximum(0, x)


def predict(params, image):
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)

batched_predict = vmap(predict, in_axes=(None, 0))

def one_hot(x, k, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)

def update(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]

def numpy_collate(batch):
    return tree_map(np.asarray, default_collate(batch))

def flatten_and_cast(pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))

mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=flatten_and_cast)
training_generator = DataLoader(mnist_dataset, batch_size=batch_size, collate_fn=numpy_collate)

train_images = mnist_dataset.data.numpy().reshape(len(mnist_dataset), -1).astype(np.float32)
train_labels = one_hot(mnist_dataset.targets.numpy(), n_targets)

mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
test_images =  mnist_dataset_test.data.numpy().reshape(len(mnist_dataset_test), -1).astype(np.float32)
test_labels = one_hot(mnist_dataset_test.targets.numpy(), n_targets)

start_time = time.time()

for epoch in range(num_epochs):
    for x, y in training_generator:
        y = one_hot(y, n_targets)
        params = update(params, x, y)

    train_acc = accuracy(params, train_images, train_labels)
    test_acc = accuracy(params, test_images, test_labels)
    print(f"train {train_acc} | test {test_acc}")

epoch_time = time.time() - start_time

print(f"Total Epoch time {epoch_time}")

