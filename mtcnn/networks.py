import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial


PNET_SRIDE = 1


def net_sequential_fn(batch, layers) -> jnp.array:
    layers = [layer(**params) for (layer, params) in layers]
    net = hk.Sequential(layers)
    return net(batch)


@jax.jit
def extend_sigmoid(x):
    return 1.5*jax.nn.sigmoid(x)

pnet_encoding_fn = partial(net_sequential_fn, layers=[
    (hk.Conv2D, dict(output_channels=10, kernel_shape=3, stride=PNET_SRIDE, padding="VALID")),
    (lambda: jax.nn.selu, dict()),
    (hk.MaxPool, dict(window_shape=2, strides=2, padding="VALID")),
    (hk.Conv2D, dict(output_channels=16, kernel_shape=3, padding="VALID")),
    (lambda: jax.nn.selu, dict()),
    (hk.Conv2D, dict(output_channels=32, kernel_shape=3, padding="VALID")),
    (lambda: jax.nn.selu, dict()),
])

rnet_encoding_fn = partial(net_sequential_fn, layers=[
    (hk.Conv2D, dict(output_channels=28, kernel_shape=3, padding="VALID")),
    (lambda: jax.nn.selu, dict()),
    (hk.MaxPool, dict(window_shape=2, strides=2, padding="VALID")),
    (hk.Conv2D, dict(output_channels=48, kernel_shape=3, padding="VALID")),
    (lambda: jax.nn.selu, dict()),
    (hk.MaxPool, dict(window_shape=2, strides=2, padding="VALID")),
    (hk.Conv2D, dict(output_channels=64, kernel_shape=2, padding="VALID")),
    (hk.Flatten, dict()),
    (lambda: jax.nn.selu, dict())
])

onet_encoding_fn = partial(net_sequential_fn, layers=[
    (hk.Conv2D, dict(output_channels=32, kernel_shape=3, padding="VALID")),
    (lambda: jax.nn.selu, dict()),
    (hk.MaxPool, dict(window_shape=2, strides=2, padding="VALID")),
    (hk.Conv2D, dict(output_channels=64, kernel_shape=3, padding="VALID")),
    (lambda: jax.nn.selu, dict()),
    (hk.MaxPool, dict(window_shape=2, strides=2, padding="VALID")),
    (hk.Conv2D, dict(output_channels=64, kernel_shape=3, padding="VALID")),
    (lambda: jax.nn.selu, dict()),
    (hk.MaxPool, dict(window_shape=2, strides=2, padding="VALID")),
    (hk.Conv2D, dict(output_channels=128, kernel_shape=3, padding="VALID")),
    (hk.Flatten, dict()),
    (lambda: jax.nn.selu, dict())
])


pnet_fc = partial(net_sequential_fn, layers=[
    (hk.Conv2D, dict(output_channels=2, kernel_shape=1, padding="VALID")),
    (lambda: jax.nn.softmax, dict())])
pnet_bbx = partial(net_sequential_fn, layers=[
    (hk.Conv2D, dict(output_channels=4, kernel_shape=1, padding="VALID"))])
pnet_fll = partial(net_sequential_fn, layers=[
    (hk.Conv2D, dict(output_channels=10, kernel_shape=1, padding="VALID"))])

rnet_fc = partial(net_sequential_fn, layers=[
    (hk.Linear, dict(output_size=2)),
    (lambda: jax.nn.softmax, dict())])
rnet_bbx = partial(net_sequential_fn, layers=[
    (hk.Linear, dict(output_size=4)),
    (lambda: extend_sigmoid, dict())])
rnet_fll = partial(net_sequential_fn, layers=[
    (hk.Linear, dict(output_size=10)),
    (lambda: extend_sigmoid, dict())])

onet_fc = partial(net_sequential_fn, layers=[
    (hk.Linear, dict(output_size=2)),
    (lambda: jax.nn.softmax, dict())])
onet_bbx = partial(net_sequential_fn, layers=[
    (hk.Linear, dict(output_size=4)),
    (lambda: extend_sigmoid, dict())])
onet_fll = partial(net_sequential_fn, layers=[
    (hk.Linear, dict(output_size=10)),
    (lambda: extend_sigmoid, dict())])

pnet_encoding_t = hk.without_apply_rng(hk.transform(pnet_encoding_fn))
pnet_fc_t = hk.without_apply_rng(hk.transform(pnet_fc))
pnet_bbx_t = hk.without_apply_rng(hk.transform(pnet_bbx))
pnet_fll_t = hk.without_apply_rng(hk.transform(pnet_fll))

rnet_encoding_t = hk.without_apply_rng(hk.transform(rnet_encoding_fn))
rnet_fc_t = hk.without_apply_rng(hk.transform(rnet_fc))
rnet_bbx_t = hk.without_apply_rng(hk.transform(rnet_bbx))
rnet_fll_t = hk.without_apply_rng(hk.transform(rnet_fll))

onet_encoding_t = hk.without_apply_rng(hk.transform(onet_encoding_fn))
onet_fc_t = hk.without_apply_rng(hk.transform(onet_fc))
onet_bbx_t = hk.without_apply_rng(hk.transform(onet_bbx))
onet_fll_t = hk.without_apply_rng(hk.transform(onet_fll))
