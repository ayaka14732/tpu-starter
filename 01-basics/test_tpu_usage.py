import jax
import jax.numpy as np
import re
import subprocess

def size_str_to_bytes(size_str: str) -> int:
    table = (
        ('GB', 1024 * 1024 * 1024),
        ('MB', 1024 * 1024),
        ('kB', 1024),  # sic, not KB
        ('B', 1),
    )
    for suffix, scale in table:
        if size_str.endswith(suffix):
            val_str = size_str[:-len(suffix)]
            val = float(val_str)
            return int(val * scale)
    raise ValueError(f'Unable to handle string {repr(size_str)}')

pattern = re.compile(r'^\s+(\S+).+?TPU_(\d+)', flags=re.MULTILINE)

def get_tpu_usage():
    jax.profiler.save_device_memory_profile('/tmp/memory.prof')
    output = subprocess.run(['go', 'tool', 'pprof', '-tags', '/tmp/memory.prof'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.decode('utf-8')
    def process_one_match(match) :
        size_str, tpu_id = match.group(1, 2)
        n_bytes = size_str_to_bytes(size_str)
        return tpu_id, n_bytes
    return list(map(process_one_match, pattern.finditer(output)))

a = np.zeros((128, 128, 128))
b = np.zeros((128, 128))

print(get_tpu_usage())
