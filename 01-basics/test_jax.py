import jax
import jax.numpy as np
import jax.random as rand

print(jax.devices())

key = rand.PRNGKey(42)

key, *subkey = rand.split(key, num=3)
a = rand.uniform(subkey[0], shape=(10000, 100000))
b = rand.uniform(subkey[1], shape=(100000, 10000))

c = np.dot(a, b)
print(c.shape)
