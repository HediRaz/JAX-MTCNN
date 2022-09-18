import jax.numpy as jnp

from mtcnn.mtcnn import SoftNMS


def test_iou_matrix():
    bbx = jnp.array([[0, 1, 2, 3], [1, 1, 2, 3], [0, 1, 1, 3]])
    res = SoftNMS.iou_matrix(3, bbx)
    print(res)
    res = jnp.triu(res, k=1)
    print(res)
    res = jnp.sum(res, 0)
    print(res)
