import jax
import jax.numpy as jnp
import time
from functools import partial

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X, y = load_iris(return_X_y=True)

# print(y)

encoder = OneHotEncoder(sparse_output=False)
y = y.reshape(-1, 1)
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# input -> h1 -> h2 -> Output
@partial(jax.jit, static_argnames=("input_dim", "hidden_dim_1", "hidden_dim_2", "output_dim"))
def init_params(input_dim, hidden_dim_1, hidden_dim_2, output_dim, random_key):
    random_keys = jax.random.split(random_key, 3)

    W1 = jax.random.normal(random_keys[0], (input_dim, hidden_dim_1))
    b1 = jnp.zeros((hidden_dim_1, ))    
    W2 = jax.random.normal(random_keys[1], (hidden_dim_1, hidden_dim_2))
    b2 = jnp.zeros((hidden_dim_2, ))    
    W3 = jax.random.normal(random_keys[2], (hidden_dim_2, output_dim))
    b3 = jnp.zeros((output_dim, ))

    return W1, b1, W2, b2, W3, b3

@jax.jit
def forward(params, X):
    W1, b1, W2, b2, W3, b3 = params

    h1 = jax.nn.relu(jnp.dot(X, W1) + b1)
    h2 = jax.nn.relu(jnp.dot(h1, W2) + b2)
    logits  = jnp.dot(h2, W3) + b3

    return logits

@jax.jit
def loss_fn(params, x, y, l2_reg=0.001):
    logits = forward(params, x)
    probs = jax.nn.softmax(logits)
    l2_loss = l2_reg * sum([jnp.sum(w ** 2) for w in params[::2]])
    return -jnp.mean(jnp.sum(y*jnp.log(probs + 1e-8), axis=1)) + l2_loss

@jax.jit
def train_step(params, x, y, lr):
    grads = jax.grad(loss_fn)(params, x, y)
    return [(param - lr * grad) for param, grad in zip(params, grads)]

@jax.jit
def accuracy(params, x, y):
    preds = jnp.argmax(forward(params, x), axis=1)
    targets = jnp.argmax(y, axis=1)
    return jnp.mean(preds == targets)

def  data_loader(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

start = time.time()
random_key = jax.random.key(1)
input_dim = X_train.shape[1]
hidden_dim_1 = 16
hidden_dim_2 = 8
output_dim = y_train.shape[1]
learning_rate = 0.005
batch_size = 16
epochs = 200

params = init_params(input_dim=input_dim, hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2, output_dim=output_dim, random_key=random_key)

for epoch in range(epochs):
    for X_batch, y_batch in data_loader(X_train, y_train, batch_size):
        params = train_step(params, X_batch, y_batch, learning_rate)

    if epoch % 10 == 0:
        train_acc = accuracy(params, X_train, y_train)
        test_acc = accuracy(params, X_test, y_test)

        print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}')
end = time.time()
print(f'Final Test Acc: {accuracy(params, X_test, y_test):.2f}')
print(f'time taken: {end - start}')
