import os
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
import jax.numpy as np
from operator import add
import subprocess

devices = jax.devices()

def show_mem(*args, **kwargs) -> str:
    jax.profiler.save_device_memory_profile('/tmp/memory.prof')
    return subprocess.run(['go', 'tool', 'pprof', '-tags', '/tmp/memory.prof'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.decode('utf-8')

def largest_v2() -> np.ndarray:
    return np.zeros((1024, 1024, 957, 2), dtype=np.float32)

def largest_v3() -> np.ndarray:
    return np.zeros((2048, 1024, 984, 2), dtype=np.float32)

print(show_mem(largest_v2()))

print(show_mem(jax.jit(largest_v2, device=devices[1])()))
print(show_mem(jax.jit(largest_v2, device=devices[2])()))
print(show_mem(jax.jit(largest_v2, device=devices[3])()))
print(show_mem(jax.jit(largest_v2, device=devices[4])()))
print(show_mem(jax.jit(largest_v2, device=devices[5])()))
print(show_mem(jax.jit(largest_v2, device=devices[6])()))
print(show_mem(jax.jit(largest_v2, device=devices[7])()))

def quarter_v2() -> np.ndarray:
    return np.zeros((256, 1024, 957, 2), dtype=np.float32)

def f() -> np.ndarray:
    a = jax.jit(quarter_v2, device=devices[1])()
    b = jax.jit(quarter_v2, device=devices[2])()
    ab = jax.jit(add, device=devices[3])(a, b)
    return ab

print(show_mem(f()))
print(show_mem(jax.jit(f, device=devices[4])()))
