import jax
import jax.lax as lax
import jax.numpy as np
import numpy as onp

# print the default devices
print('jax.devices():', jax.devices())

# define functions that convert between string and array
str2arr = lambda s: onp.array(list(s.encode('utf-8')))
arr2str = lambda a: bytes(a.tolist()).decode('utf-8')

# define our strings
a = '你好👋，'
b = '世界🌎！'

# convert strings to array
array_a = str2arr(a)
array_b = str2arr(b)
print('array_a:', repr(array_a))
print('array_b:', repr(array_b))

# transfer arrays to tpu
array_a = np.asarray(array_a)
array_b = np.asarray(array_b)
print('array_a:', repr(array_a))
print('array_b:', repr(array_b))
print('array_a.device_buffer.device():', repr(array_a.device_buffer.device()))
print('array_b.device_buffer.device():', repr(array_b.device_buffer.device()))

# concatenate two arrays on tpu
array_ab = lax.concatenate((array_a, array_b), 0)
print('array_ab:', repr(array_ab))
print('array_ab.device_buffer.device():', repr(array_ab.device_buffer.device()))

# transfer result back to cpu and convert back to string
ab = arr2str(array_ab)
print('ab:', ab)
