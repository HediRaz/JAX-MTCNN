import cv2
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from PIL import Image
from tqdm import tqdm

from train.utils_dataset import load_labels, iou_xyhw
from tests.jittable_test import jittable_test
from mtcnn.mtcnn import SoftNMS, MTCNN, floor_2powlog2
from mtcnn.utils import draw_box, draw_fll


def test_floor2powlog2():
    jittable_test(floor_2powlog2, lambda key: jax.random.randint(key, (1,), 1, 5000))
    for n in [1, 2, 4, 8, 16, 32, 64, 128]:
        assert floor_2powlog2(n) == n
        if n != 1:
            assert floor_2powlog2(n+1) == n


def test_iou_matrix():
    bbx = jnp.array([[0, 1, 2, 3], [1, 1, 1, 3], [0, 1, 1, 3]])
    res = SoftNMS.iou_matrix(3, bbx)
    assert jnp.array_equal(res, jnp.array([[1, 0.5, 0.5], [0.5, 1, 0], [0.5, 0, 1]]))
    res = jnp.triu(res, k=1)
    assert jnp.array_equal(res, jnp.array([[0, 0.5, 0.5], [0, 0, 0], [0, 0, 0]]))
    res = jnp.sum(res, 0)
    assert jnp.array_equal(res, jnp.array([0, 0.5, 0.5]))


def test_get_best_preds():
    mtcnn = MTCNN()
    jittable_test(MTCNN.get_preds_pnet_1, lambda key: (
        16800,
        0.7,
        jax.random.uniform(key, (120, 140, 2)),
        jax.random.randint(key, (120, 140, 4), 0, 120),
    ))
    jittable_test(MTCNN.get_preds_pnet_2, lambda key: (
        int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))),
        jax.random.uniform(key, (120*140, 2)),
        jax.random.randint(key, (120*140, 4), 0, 120),
        mtcnn.idx_mask_x,
        mtcnn.idx_mask_y,
        1.
        ), _print=True)
    jittable_test(mtcnn.get_preds_pnet, lambda key: (
        jax.random.uniform(key, (120, 140, 2)),
        jax.random.randint(key, (120, 140, 4), 0, 120),
        1.,
        (124*2+2, 144*2+2)
        ), _print=True)

    mtcnn = MTCNN()

    labels = load_labels("datasets/WIDER/WIDER_train/labels.json")
    img_filename = list(labels.keys())[0]
    img_dict = labels[img_filename]
    img_bbx = np.array(img_dict["bbx"], dtype=np.float32)
    print(img_bbx)
    img = np.array(Image.open("datasets/WIDER/WIDER_train/images/"+img_filename))
    o_img_shape = img.shape
    print(o_img_shape)
    img = cv2.resize(img, (120, 100), interpolation=cv2.INTER_CUBIC)
    img_bbx[:, ::2] *= 100 / o_img_shape[0]
    img_bbx[:, 1::2] *= 120 / o_img_shape[1]
    print(img_bbx)
    img_shape = img.shape
    print(img_shape)
    res_shape = (img_shape[0]-2)//2-4, (img_shape[1]-2)//2-4

    fc = np.zeros((res_shape[0], res_shape[1], 2))
    bbx = np.zeros((res_shape[0], res_shape[1], 4))
    for i in tqdm(range(res_shape[0])):
        for j in range(res_shape[1]):
            box = np.array([2*i, 2*j, 12, 12])
            ious = [iou_xyhw(box, b) for b in img_bbx]
            if max(ious) > 0.65:
                fc[i, j, 0] = 1
                bbx[i, j] = np.array([0, 0, 1, 1])
            else:
                fc[i, j, 1] = 1

    print(np.sum(fc[:, :, 0]))

    fc, bbx = mtcnn.get_preds_pnet(fc, bbx, 1, img_shape)
    print(len(fc))
    print(fc, bbx)
    # bbx = np.array(bbx)

    for box in bbx:
        # img = draw_box(img, [box[1], box[0], box[2], box[3]])
        img = draw_box(img, box)

    plt.imshow(img)
    plt.savefig("results/res.png")


def test_pnet_inference():
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    mtcnn = MTCNN(subkey)

    jittable_test(mtcnn.pnet_inference, lambda key: (
        jax.random.uniform(key, (120, 140, 3)),
        1
        ), _print=True)

    img = jnp.ones((120, 140, 3), dtype=jnp.float32)
    res = mtcnn.pnet_inference(img, 1.)
    assert res[0].shape == ((118//2 - 4)*(138//2 - 4), 2)
    assert res[1].shape == ((118//2 - 4)*(138//2 - 4), 4)

    mtcnn.pnet_threshold = 1.
    res = mtcnn.pnet_inference(img, 1.)
    assert res[0].shape == (0, 2)
    assert res[1].shape == (0, 4)


def test_rnet_inference():
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    mtcnn = MTCNN(subkey)

    # jittable_test(MTCNN.rnet_inference_1, lambda key: (
    #     int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))),
    #     jax.random.randint(key, (int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))), 4), 0, 120)
    #     ), _print=True)

    res = MTCNN.rnet_inference_1((120, 140), jax.random.randint(subkey, (256, 4), 1, 120))
    assert res[0].shape == (256,)
    assert res[1].shape == (256,)
    assert res[2].shape == (256,)
    assert jnp.sum(jnp.where(res[2] < 1, 1, 0)) == 0

    # jittable_test(MTCNN.rnet_inference_2, lambda key: (
    #     jax.random.uniform(key, (120, 140, 3)),
    #     jax.random.randint(key, (int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))),), 0, 120),
    #     jax.random.randint(key, (int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))),), 0, 110),
    #     jax.random.randint(key, (int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))),), 1, 60)
    #     ), _print=True)

    res = MTCNN.rnet_inference_2(
        jax.random.uniform(subkey, (120, 140, 3)),
        jax.random.randint(subkey, (256,), 0, 120),
        jax.random.randint(subkey, (256,), 0, 120),
        jax.random.randint(subkey, (256,), 1, 120),
    )
    assert res.shape == (256, 24, 24, 3)

    # jittable_test(MTCNN.rnet_inference_3, lambda key: (
    #     int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))),
    #     mtcnn.rnet_params,
    #     jax.random.uniform(key, (int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))), 24, 24, 3)),
    #     mtcnn.rnet_threshold
    #     ), _print=True)

    res = MTCNN.rnet_inference_3(
        int(floor_2powlog2(jax.random.randint(subkey, (1,), 1, 5000))),
        mtcnn.rnet_params,
        jax.random.uniform(subkey, (int(floor_2powlog2(jax.random.randint(subkey, (1,), 1, 5000))), 24, 24, 3)),
        mtcnn.rnet_threshold
        )
    n = int(floor_2powlog2(jax.random.randint(subkey, (1,), 1, 5000)))
    assert res[0].shape == (n, 2)
    assert res[1].shape == (n, 4)
    assert res[2].shape == (n, 10)

    # jittable_test(MTCNN.rnet_inference_4, lambda key: (
    #     int(floor_2powlog2(jax.random.randint(key, (1,), 2, 5000))),
    #     int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000)))//2,
    #     jax.random.uniform(key, (int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))), 2)),
    #     jax.random.randint(key, (int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))), 4), 1, 120),
    #     jax.random.randint(key, (int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))), 4), 1, 120),
    #     jax.random.randint(key, (int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))), 10), 1, 120),
    #     jax.random.randint(key, (int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))),), 1, 120),
    #     ), num_iter=40, _print=True)

    res = MTCNN.rnet_inference_4(
        int(floor_2powlog2(jax.random.randint(subkey, (1,), 2, 5000))),
        int(floor_2powlog2(jax.random.randint(subkey, (1,), 1, 5000)))//2,
        jax.random.uniform(subkey, (int(floor_2powlog2(jax.random.randint(subkey, (1,), 1, 5000))), 2)),
        jax.random.randint(subkey, (int(floor_2powlog2(jax.random.randint(subkey, (1,), 1, 5000))), 4), 1, 120),
        jax.random.randint(subkey, (int(floor_2powlog2(jax.random.randint(subkey, (1,), 1, 5000))), 4), 1, 120),
        jax.random.randint(subkey, (int(floor_2powlog2(jax.random.randint(subkey, (1,), 1, 5000))), 10), 1, 120),
        jax.random.randint(subkey, (int(floor_2powlog2(jax.random.randint(subkey, (1,), 1, 5000))),), 1, 120),
    )
    new_n = int(floor_2powlog2(jax.random.randint(subkey, (1,), 1, 5000)))//2
    assert res[0].shape == (new_n, 2)
    assert res[1].shape == (new_n, 4)
    assert res[2].shape == (new_n, 4)
    assert res[3].shape == (new_n, 10)
    assert res[4].shape == (new_n,)

    # jittable_test(MTCNN.rnet_inference_5, lambda key: (
    #     int(floor_2powlog2(jax.random.randint(key, (1,), 2, 5000))),
    #     jax.random.randint(key, (int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))), 4), 1, 120),
    #     jax.random.randint(key, (int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))), 4), 1, 120),
    #     jax.random.randint(key, (int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))), 10), 1, 120),
    #     jax.random.randint(key, (int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))),), 1, 120),
    #     ), num_iter=20, _print=True)

    # jittable_test(mtcnn.rnet_inference, lambda key: (
    #     jax.random.uniform(key, (120, 140, 3)),
    #     jax.random.randint(key, (int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))), 4), 1, 110),
    #     ), num_iter=20, _print=True)


    labels = load_labels("datasets/CELEBA/labels.json")
    img_filename = list(labels.keys())[0]
    img_dict = labels[img_filename]
    img_box = np.array(img_dict["box"], dtype=np.float32)
    img_fll = np.array(img_dict["fll"], dtype=np.float32)
    print(img_box)
    print(img_fll)
    img = np.array(Image.open("datasets/CELEBA/img_celeba/"+img_filename))
    o_img_shape = img.shape
    print(o_img_shape)
    img = cv2.resize(img, (45, 60), interpolation=cv2.INTER_CUBIC)
    img_box[::2] *= 60 / o_img_shape[0]
    img_box[1::2] *= 45 / o_img_shape[1]
    img_fll[::2] *= 60 / o_img_shape[0]
    img_fll[1::2] *= 45 / o_img_shape[1]
    print(img_box)
    img_shape = img.shape
    print(img_shape)
    res_shape = (img_shape[0]-2)//2-4 - 12, (img_shape[1]-2)//2-4 - 12

    fc = np.zeros((res_shape[0], res_shape[1], 2))
    bbx = np.zeros((res_shape[0], res_shape[1], 4))
    fll = np.zeros((res_shape[0], res_shape[1], 10))
    bbx_pred = np.zeros((res_shape[0], res_shape[1], 4))
    fll_pred = np.zeros((res_shape[0], res_shape[1], 10))
    for i in tqdm(range(res_shape[0])):
        for j in range(res_shape[1]):
            box = np.array([2*i, 2*j, 24, 24])
            iou = iou_xyhw(box, img_box)
            if iou > 0.65:
                fc[i, j, 0] = 1
                bbx[i, j] = np.array(img_box)
                bbx_pred[i, j] = np.array([(img_box[0]-box[0])/24, (img_box[1]-box[2])/24, img_box[2]/24, img_box[3]/24])
                fll[i, j] = np.array(img_fll)
                fll_pred[i, j] = np.array([(img_fll[i]-img_box[i%2])/24 for i in range(10)])
            else:
                fc[i, j, 1] = 1

    print(np.sum(fc[:, :, 0]))

    # fc, bbx = mtcnn.get_preds_pnet(fc, bbx, 1, img_shape)
    mask = np.where(fc[:, :, 0] > 0.5)
    fc = fc[mask]
    bbx = bbx[mask]
    fll = fll[mask]
    bbx_pred = bbx_pred[mask]
    fll_pred = fll_pred[mask]
    hw = np.ones((len(bbx_pred),))*24

    bbx_pred, fll_pred = MTCNN.rnet_inference_5(len(bbx_pred), bbx, bbx_pred, fll_pred, hw)
    print(len(fc))
    # bbx = np.array(bbx)

    # for box, f in zip(bbx, fll):
    for box, f in zip(bbx_pred, fll_pred):
        # img = draw_box(img, [box[1], box[0], box[2], box[3]])
        img = draw_box(img, np.int32(box))
        img = draw_fll(img, np.int32(f))

    plt.imshow(img)
    plt.savefig("results/res.png")


def test_onet_inference():
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    mtcnn = MTCNN(subkey)

    jittable_test(mtcnn.onet_inference, lambda key: (
        jax.random.uniform(key, (120, 140, 3)),
        jax.random.randint(key, (int(floor_2powlog2(jax.random.randint(key, (1,), 1, 5000))), 4), 1, 110),
        ), num_iter=20, _print=True)


def test_mtcnn_call():
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    mtcnn = MTCNN(subkey)

    img = jax.random.uniform(subkey, (128, 140, 3))
    res = mtcnn(img)
    print(res[0].shape)
    print(res[1].shape)
    print(res[2].shape)

    from time import time
    subkeys = jax.random.split(key, 100)
    for i in range(100):
        print('#'*30)
        img = jax.random.uniform(subkeys[i], (128, 140, 3))
        top = time()
        mtcnn(img)
        print(time() - top, "second")
        print(1/(time() - top), "image per second")
