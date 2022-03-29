import jax
import jax.numpy as np
import jax.random as rand

@jax.jit
def complex_operation(a, b):
    c = np.einsum('abc,cdb->ad', a, b)
    d = a.sum(axis=-1)
    e = np.sin(c).T @ d
    return np.sum(e)

def single_core(key):
    k1, k2 = rand.split(key)
    a = rand.normal(k1, shape=(1000, 401, 200))
    b = rand.normal(k2, shape=(200, 3451, 401))
    c = complex_operation(a, b)
    return np.sum(c)

def f(key):
    keys = rand.split(key, num=8)
    c = jax.pmap(single_core)(keys)
    return np.sum(c)

with jax.profiler.trace('/tmp/jax-profiler'):
    key = rand.PRNGKey(42)
    for _ in range(10):
        key, subkey = rand.split(key)
        m = f(subkey)
        print(m)

# View with `tensorboard --logdir=/tmp/jax-profiler`
