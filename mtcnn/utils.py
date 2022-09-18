from typing import Sequence, Tuple

import jax.numpy as jnp
import numpy as np
import cv2


def to_xyxy(bbx: jnp.ndarray) -> jnp.ndarray:
    bbx = bbx.at[2].set(bbx.at[0].get() + bbx.at[2].get())
    bbx = bbx.at[3].set(bbx.at[1].get() + bbx.at[3].get())
    return bbx


def to_xyhw(bbx: jnp.ndarray) -> jnp.ndarray:
    bbx = bbx.at[2].set(bbx.at[2].get() - bbx.at[0].get())
    bbx = bbx.at[3].set(bbx.at[3].get() - bbx.at[1].get())
    return bbx


def iou_xyxy(bbx1: jnp.ndarray, bbx2: jnp.ndarray) -> float:
    """ Compute IOU of two bbx [x1, y1, x2, y2] """
    h = jnp.maximum(0, jnp.minimum(bbx1[2], bbx2[2]) - jnp.maximum(bbx1[0], bbx2[0]))
    w = jnp.maximum(0, jnp.minimum(bbx1[3], bbx2[3]) - jnp.maximum(bbx1[1], bbx2[1]))
    a1 = (bbx1[2] - bbx1[0]) * (bbx1[3] - bbx1[1])
    a2 = (bbx2[2] - bbx2[0]) * (bbx2[3] - bbx2[1])
    inter = h*w
    return inter / (a1 + a2 - inter)


def iou_xyhw(bbx1: jnp.ndarray, bbx2: jnp.ndarray) -> float:
    """ Compute IOU of two bbx [x, y, h, w] """
    h = jnp.maximum(0, jnp.minimum(bbx1[0]+bbx1[2], bbx2[0]+bbx2[2]) - jnp.maximum(bbx1[0], bbx2[0]))
    w = jnp.maximum(0, jnp.minimum(bbx1[1]+bbx1[3], bbx2[1]+bbx2[3]) - jnp.maximum(bbx1[1], bbx2[1]))
    a1 = bbx1[2] * bbx1[3]
    a2 = bbx2[2] * bbx2[3]
    inter = h*w
    return inter / (a1 + a2 - inter)


def draw_box(img: np.ndarray, box: Sequence[int], color: Tuple[int] = (255, 0, 0)):
    img = cv2.rectangle(img, (box[1], box[0]), (box[1]+box[3], box[0]+box[2]), color=color, thickness=1)
    return img


def draw_fll(img: np.ndarray, fll: Sequence[int], color: Tuple[int] = (255, 0, 0)):
    for i in range(5):
        img = cv2.circle(img, (fll[2*i+1], fll[2*i]), radius=1, color=color, thickness=2)
    return img
