import tensorflow as tf
from tensorflow.python.client import device_lib

print('Tensorflow version:', tf.__version__)

@tf.function
def add_fn(x, y):
    z = x + y
    return z

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.tpu.experimental.initialize_tpu_system(resolver)

print(device_lib.list_local_devices())
