import os
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=8'

import jax
jax.config.update('jax_platform_name', 'cpu')

from functools import partial
import jax.numpy as np
import jax.random as rand

devices = jax.devices()
n_devices = jax.device_count()
assert n_devices == 8

batch_size = 3 * n_devices
n_epochs = 5
key = rand.PRNGKey(42)

@partial(jax.pmap, axis_name='num_devices')
def update(params, a, b):
    delta = np.mean(a @ b, axis=0)
    delta = jax.lax.pmean(delta, axis_name='num_devices')  # calculate mean across devices
    new_params = params + delta
    return new_params

key, subkey = rand.split(key)
params = rand.uniform(subkey, (4, 4))
replicated_params = jax.device_put_replicated(params, devices)

for _ in range(n_epochs):
    key, *subkey = rand.split(key, num=3)
    a = rand.uniform(subkey[0], (n_devices, batch_size // n_devices, 4, 1))
    b = rand.uniform(subkey[1], (n_devices, batch_size // n_devices, 1, 4))
    replicated_params = update(replicated_params, a, b)

params = jax.tree_map(lambda x: x[0], replicated_params)
print(params.shape)  # (4, 4)
