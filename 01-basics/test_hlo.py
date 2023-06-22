# TODO: The code no longer works. The new simplified approach is described in
# https://github.com/google/jax/discussions/7068#discussioncomment-5442809

import jax
import jax.lax as lax
from jax.lib import xla_bridge
import jax.numpy as np
from jaxlib.xla_extension import HloPrintOptions

# jax.config.update('jax_platforms', 'cpu')

backend = xla_bridge.get_backend()
print(backend.platform_version)

option = HloPrintOptions()
# option.print_metadata = False
# option.include_layout_in_shapes = False
# option.print_extra_attributes = False

@jax.xla_computation
def f(x, y):
    a = np.einsum('pqrs,tuqvr->pstuv', x, y)
    return lax.sin(a)

c = f(np.ones((3,4,5,6)), np.ones((7,8,4,9,5)))
module = backend.compile(c).hlo_modules()[0]
hlo_text = module.to_string(option)
print(hlo_text)

with open('/tmp/hlo.txt', 'w') as f:
    print(hlo_text, file=f)
