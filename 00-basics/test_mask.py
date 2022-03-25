# `np.NINF` is a common trick if you want to 'zero out' something in the softmax function

import jax.nn as nn
import jax.numpy as np

a = np.array([[[3.2, 1.4, 0.5, 2.3, 4.3],
               [7.5, 0.6, 5.9, 0.9, 7.0],
               [0.4, 0.9, 7.1, 7.0, 3.2],
               [6.4, 2.1, 0.7, 0.6, 2.8],
               [0.8, 0.6, 0.5, 7.5, 2.3]],
              [[3.2, 1.4, 0.5, 2.3, 4.3],
               [7.5, 0.6, 5.9, 0.9, 7.0],
               [0.4, 0.9, 7.1, 7.0, 3.2],
               [6.4, 2.1, 0.7, 0.6, 2.8],
               [0.8, 0.6, 0.5, 7.5, 2.3]]])

b = np.array([[1, 1, 1, 0, 0],
              [1, 1, 0, 0, 0]], dtype=np.bool_)
mask = np.einsum('bi,bj->bij', b, b)
print('mask:', mask)

'''
This is equal to:

mask = np.array([[[1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]],
                 [[1, 1, 0, 0, 0],
                  [1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]], dtype=np.bool_)
'''

c = np.where(mask, a, np.NINF)
print('c:', c)

d = nn.softmax(c)
print('d:', d)

e = np.where(mask, d, 0.)
print('e:', e)

f = e.sum(-1)
print('f:', f)
