### 9.2. About JAX

#### 9.2.1. Indexing an array with an array

```python
import jax.numpy as np
import numpy as onp

a = onp.arange(12).reshape((6, 2))
b = onp.arange(6).reshape((2, 3))

a_ = np.asarray(a)
b_ = np.asarray(b)

a[b]  # success
a_[b_]  # success
a_[b]  # success
a[b_]  # error: index 3 is out of bounds for axis 1 with size 2
```

Generally speaking, JAX supports NumPy arrays, but NumPy does not support JAX arrays.

#### 9.2.2. `np.dot` and `torch.dot` are different

```python
import numpy as onp
import torch

a = onp.random.rand(3, 4, 5)
b = onp.random.rand(4, 5, 6)
onp.dot(a, b)  # success

a_ = torch.from_numpy(a)
b_ = torch.from_numpy(b)
torch.dot(a_, b_)  # error: 1D tensors expected, but got 3D and 3D tensors
```

#### 9.2.3. `np.std` and `torch.std` are different

```python
import torch

x = torch.tensor([[-1., 1.]])

print(x.std(-1).numpy())  # [1.4142135]
print(x.numpy().std(-1))  # [1.]
```

This is because in [`np.std`](https://numpy.org/doc/stable/reference/generated/numpy.std.html) the denominator is _n_, while in [`torch.std`](https://pytorch.org/docs/stable/generated/torch.std.html) it is _n_-1. See [pytorch/pytorch#1854](https://github.com/pytorch/pytorch/issues/1854) for details.

#### 9.2.4. Computations on TPU are in low precision by default

JAX uses bfloat16 for matrix multiplication on TPU by default, even if the data type is float32.

```python
import jax.numpy as np

print(4176 * 5996)  # 25039296

a = np.array(0.4176, dtype=np.float32)
b = np.array(0.5996, dtype=np.float32)
print((a * b).item())  # 0.25039297342300415
```

To do matrix multiplication in float32, you need to add this line at the top of the script:

```python
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
```

Other precision values can be found in [jax.lax.Precision](https://jax.readthedocs.io/en/latest/jax.lax.html#jax.lax.Precision). See [google/jax#9973](https://github.com/google/jax/issues/9973) for details.

#### 9.2.5. Weight matrix of linear layer is transposed in PyTorch

Weight matrix of linear layer is transposed in PyTorch, but not in Flax. Therefore, if you want to convert model parameters between PyTorch and Flax, you needed to transpose the weight matrices.

In Flax:

```python
import flax.linen as nn
import jax.numpy as np
import jax.random as rand
linear = nn.Dense(5)
key = rand.PRNGKey(42)
params = linear.init(key, np.zeros((3,)))
print(params['params']['kernel'].shape)  # (3, 5)
```

In PyTorch:

```python
import torch.nn as nn
linear = nn.Linear(3, 5)
print(linear.weight.shape)  # (5, 3), not (3, 5)
```
