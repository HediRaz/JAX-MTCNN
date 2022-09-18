import jax
import jax.numpy as jnp
from mtcnn.utils import to_xyhw, to_xyxy, iou_xyhw, iou_xyxy
from tests.jittable_test import jittable_test


def test_bbx_format_change():
    bbx = jnp.array([0, 1, 5, 3])
    assert jnp.array_equal(to_xyxy(bbx), jnp.array([0, 1, 5, 4]))
    assert jnp.array_equal(to_xyhw(bbx), jnp.array([0, 1, 5, 2]))

    jittable_test(jax.jit(to_xyxy), lambda key: (jax.random.randint(key, (4,), 0, 40),))
    jittable_test(jax.jit(to_xyhw), lambda key: (jax.random.randint(key, (4,), 0, 40),))


def test_iou():
    bbx1 = [0, 0, 4, 4]
    bbx2 = [2, 1, 3, 3]
    assert iou_xyhw(bbx1, bbx1) == iou_xyxy(bbx1, bbx1) == 1
    assert iou_xyxy(bbx2, bbx1) == iou_xyxy(bbx1, bbx2) == 2 / (4*4)
    assert iou_xyhw(bbx2, bbx1) == iou_xyhw(bbx1, bbx2) == (2*3) / (4*4+3)

    bbx1 = [0, 0, 2, 2]
    bbx2 = [0, 1, 2, 1]
    assert iou_xyhw(bbx1, bbx1) == iou_xyxy(bbx1, bbx1) == 1
    assert iou_xyxy(bbx2, bbx1) == iou_xyxy(bbx1, bbx2) == 0
    assert iou_xyhw(bbx2, bbx1) == iou_xyhw(bbx1, bbx2) == 1 / 2

    jittable_test(jax.jit(iou_xyxy),
                  lambda key: (jax.random.randint(jax.random.split(key)[0], (4,), 0, 40), jax.random.randint(jax.random.split(key)[1], (4,), 0, 40)))
    jittable_test(jax.jit(iou_xyhw),
                  lambda key: (jax.random.randint(jax.random.split(key)[0], (4,), 0, 40), jax.random.randint(jax.random.split(key)[1], (4,), 0, 40)))
