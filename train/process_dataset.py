import os
import json
import random as rd
import numpy as np
import jax.numpy as jnp
from PIL import Image
from math import log2
from tqdm import tqdm

from train.utils_dataset import load_labels, iou_xyhw
from mtcnn.mtcnn import MTCNN, Pyramid


WIDER_DIR = "datasets/WIDER"
WIDER_TRAIN_DIR = "datasets/WIDER/WIDER_train"
CELEBA_DIR = "datasets/CELEBA"


def process_wider_labels():
    labels = dict()
    with open(os.path.join(WIDER_TRAIN_DIR, "wider_face_train_bbx_gt.txt"), "r") as labels_file:
        lines = labels_file.readlines()
        while len(lines) > 0:
            img_filename = lines.pop(0).rstrip()
            nb_face = int(lines.pop(0))
            if nb_face == 0:
                lines.pop(0)
                continue
            labels[img_filename] = dict(filename=img_filename, nb_face=nb_face, bbx=[])

            for _ in range(nb_face):
                bbx = lines.pop(0).split()
                labels[img_filename]["bbx"].append([int(bbx[1]), int(bbx[0]), int(bbx[3]), int(bbx[2])])

    with open(os.path.join(WIDER_TRAIN_DIR, "labels.json"), "w") as processed_labels_file:
        json.dump(labels, processed_labels_file)


def process_celeba_labels():
    labels = dict()
    with open(os.path.join(CELEBA_DIR, "list_bbox_celeba.txt"), "r") as bbx_file:
        with open(os.path.join(CELEBA_DIR, "list_landmarks_celeba.txt"), "r") as fll_file:
            bbx_lines = bbx_file.readlines()[2:]
            fll_lines = fll_file.readlines()[2:]
            for b_line, f_line in zip(bbx_lines, fll_lines):
                img_filename = b_line.split()[0]
                box = tuple([int(e) for e in b_line.split()[1:]])
                box = (box[1], box[0], box[3], box[2])
                fll = [int(e) for e in f_line.split()[1:]]
                fll = (fll[1], fll[0], fll[3], fll[2], fll[5], fll[4], fll[7], fll[6], fll[9], fll[8])
                labels[img_filename] = dict(box=box, fll=fll)

    with open(os.path.join(CELEBA_DIR, "labels.json"), "w") as processed_labels_file:
        json.dump(labels, processed_labels_file)


def generate_negative_sample(img_shape, bbx):
    neg_bbx = []
    sample_bbx = bbx[np.random.randint(0, len(bbx), max(1, (10-len(bbx)),), dtype=np.int32)]
    for box in sample_bbx:
        size = max(box[2], box[3])
        size = int(np.random.uniform(0.5, 2.)*size)
        size = max(12, size)
        random_box = np.empty((4,), dtype=np.int32)
        if img_shape[0]-size <= 0 or img_shape[1] - size <= 0:
            continue
        random_box[0] = np.random.randint(0, img_shape[0]-size, dtype=np.int32)
        random_box[1] = np.random.randint(0, img_shape[1]-size, dtype=np.int32)
        random_box[2] = size
        random_box[3] = size
        if np.all(np.array(list(map(lambda b: iou_xyhw(b, random_box), bbx))) < 0.3):
            neg_bbx.append(random_box)
    if len(neg_bbx) == 0:
        return np.zeros((0, 4))
    return np.stack(neg_bbx, 0)


def generate_positive_sample(img_shape, bbx, fll=None):
    sample_bbx = np.concatenate((bbx, bbx), 0)
    pos_bbx = []
    bbx_label = []
    for box in sample_bbx:
        size = max(box[2], box[3])
        size = int(np.random.uniform(0.8, 1.3)*size)
        size = max(12, size)
        random_box = np.empty((4,), dtype=np.int32)

        low = max(0, box[0]-int(size*0.3))
        high = min(box[0]+int(size*0.3), img_shape[0]-size)
        if low >= high:
            continue
        random_box[0] = np.random.randint(low, high, dtype=int)

        low = max(0, box[1]-int(size*0.3))
        high = min(box[1]+int(size*0.3), img_shape[1]-size)
        if low >= high:
            continue
        random_box[1] = np.random.randint(low, high, dtype=int)

        random_box[2] = size
        random_box[3] = size
        if iou_xyhw(box, random_box) > 0.65:
            pos_bbx.append(random_box)
            box_label = np.array([(box[0]-random_box[0])/size, (box[1]-random_box[1])/size, box[2]/size, box[3]/size])
            bbx_label.append(box_label)
    if len(pos_bbx) == 0:
        return np.zeros((0, 4)), np.zeros((0, 4))
    return np.stack(pos_bbx, 0), np.stack(bbx_label, 0)


def generate_pnet_dataset():
    labels = load_labels(os.path.join(WIDER_TRAIN_DIR, "labels.json"))
    pnet_labels = dict(neg=[], pos=[])

    for img_filename in tqdm(labels):
        img = Image.open(os.path.join(os.path.join(WIDER_TRAIN_DIR, "images"), img_filename))
        img_shape = np.array(img).shape
        neg_sample = generate_negative_sample(img_shape, np.array(labels[img_filename]["bbx"]))
        pos_sample = generate_positive_sample(img_shape, np.array(labels[img_filename]["bbx"]))
        for i in range(len(neg_sample)):
            pnet_labels["neg"].append((
                img_filename,
                (0, 1),
                (0, 0, 0, 0),
                (int(neg_sample[i][0]), int(neg_sample[i][1]), int(neg_sample[i][2]), int(neg_sample[i][3]))
            ))
        for i in range(len(pos_sample[0])):
            pnet_labels["pos"].append((
                img_filename,
                (1, 0),
                (pos_sample[1][i][0], pos_sample[1][i][1], pos_sample[1][i][2], pos_sample[1][i][3]),
                (int(pos_sample[0][i][0]), int(pos_sample[0][i][1]), int(pos_sample[0][i][2]), int(pos_sample[0][i][3]))
            ))

    with open(os.path.join(WIDER_TRAIN_DIR, "pnet_labels.json"), "w") as pnet_labels_file:
        json.dump(pnet_labels, pnet_labels_file)


def get_img_size_crop(img_shape):
    hw = min(img_shape[0], img_shape[1])
    hw = int(2**int(log2(hw)))
    return hw


def compute_pnet_preds():
    mtcnn = MTCNN()
    mtcnn.pnet_threshold = 0.4
    labels = load_labels(os.path.join(WIDER_TRAIN_DIR, "labels.json"))

    for img_filename in tqdm(labels):
        img = Image.open(os.path.join(WIDER_TRAIN_DIR, "images", img_filename))
        img = np.array(img)
        hw = get_img_size_crop(img.shape)
        img = img[:hw, :hw] / 255
        p_factors = Pyramid.compute_scale_factors(mtcnn.pyramid.scale_factor, img.shape)
        fc, bbx = [], []
        for factor in np.array(p_factors):
            resized_img = Pyramid.resize_image(factor, img.shape, img)
            fc_i, bbx_i = mtcnn.pnet_inference(resized_img, factor)
            fc.append(fc_i)
            bbx.append(bbx_i)
        del fc_i
        del bbx_i
        fc = jnp.concatenate(fc, 0)
        bbx = jnp.concatenate(bbx, 0)
        if len(fc) > 0:
            new_n = int(2**int(log2(len(fc))))
            fc, bbx = MTCNN.pnet_prune(new_n, fc, bbx)
            fc, bbx, _ = mtcnn.nms(fc, bbx, bbx, 0.7)

        fc = np.array(fc)
        bbx = np.array(bbx)

        yield img_filename, img.shape, fc, bbx

        # labels[img_filename]["pnet_fc_preds"] = []
        # labels[img_filename]["pnet_bbx_preds"] = []
        # for fc_pred, box in zip(fc, bbx):
        #     labels[img_filename]["pnet_fc_preds"].append((fc_pred[0], fc_pred[1]))
        #     labels[img_filename]["pnet_bbx_preds"].append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))

    # with open(os.path.join(WIDER_TRAIN_DIR, "pnet_preds.json"), "w") as pnet_preds_file:
    #     json.dump(labels, pnet_preds_file)


def get_box_label(box, bbx):
    for true_box in bbx:
        iou = iou_xyhw(box, true_box)
        if iou > 0.65:
            return ("pos", True, True, (1, 0),
                    ((true_box[0]-box[0])/box[2], (true_box[1]-box[1])/box[2], true_box[2]/box[2], true_box[3]/box[2]))
        elif iou > 0.3:
            return ("partial", False, True, (1, 0),
                    ((true_box[0]-box[0])/box[2], (true_box[1]-box[1])/box[2], true_box[2]/box[2], true_box[3]/box[2]))
    return ("neg", True, False, (0, 1), (0, 0, 0, 0))


def generate_rnet_dataset():
    labels = load_labels(os.path.join(WIDER_TRAIN_DIR, "labels.json"))
    rnet_labels = []
    count = dict(pos=0, neg=0, partial=0)
    c = 0

    for img_filename, img_shape, fc_pred, bbx_pred in compute_pnet_preds():
        c += 1
        for box in bbx_pred:
            mid_x = (2*box[0] + box[2])/2
            mid_y = (2*box[1] + box[3])/2
            hw = min(img_shape[0]-2, img_shape[1]-2, max(1, max(box[2], box[3])*1.1))
            x0 = int(min(img_shape[0]-hw-1, max(0, int(mid_x - hw/2))))
            y0 = int(min(img_shape[1]-hw-1, max(0, int(mid_y - hw/2))))
            hw = int(hw)
            box = (x0, y0, hw, hw)
            kind, mask_fc, mask_bbx, box_fc, box_label = get_box_label(box, labels[img_filename]["bbx"])

            if kind == "partial" and count[kind] <= min(list(count.values())) + 3:
                count[kind] += 1
                rnet_labels.append((img_filename, mask_fc, mask_bbx, box_fc, box_label, box))
            elif count[kind] <= 3*min(list(count.values())) + 3:
                count[kind] += 1
                rnet_labels.append((img_filename, mask_fc, mask_bbx, box_fc, box_label, box))

        if c % 100 == 0:
            print(c, count)
            with open(os.path.join(WIDER_TRAIN_DIR, f"rnet_labels.json"), "w") as pnet_labels_file:
                json.dump(rnet_labels, pnet_labels_file)


def compute_rnet_preds():
    mtcnn = MTCNN()
    mtcnn.pnet_threshold = 0.8
    labels = load_labels(os.path.join(WIDER_TRAIN_DIR, "labels.json"))

    for img_filename in tqdm(labels):
        img = Image.open(os.path.join(WIDER_TRAIN_DIR, "images", img_filename))
        img = np.array(img)
        hw = get_img_size_crop(img.shape)
        img = img[:hw, :hw] / 255
        p_factors = Pyramid.compute_scale_factors(mtcnn.pyramid.scale_factor, img.shape)
        fc, bbx = [], []
        for factor in np.array(p_factors):
            resized_img = Pyramid.resize_image(factor, img.shape, img)
            fc_i, bbx_i = mtcnn.pnet_inference(resized_img, factor)
            fc.append(fc_i)
            bbx.append(bbx_i)
        del fc_i
        del bbx_i
        fc = jnp.concatenate(fc, 0)
        bbx = jnp.concatenate(bbx, 0)
        if len(fc) > 0:
            new_n = int(2**int(log2(len(fc))))
            fc, bbx = MTCNN.pnet_prune(new_n, fc, bbx)
            fc, bbx, _ = mtcnn.nms(fc, bbx, bbx, 0.4)

        if len(fc) != 0:
            fc, bbx, _ = mtcnn.rnet_inference(img, bbx)
            fc, bbx, _ = mtcnn.nms(fc, bbx, bbx, 0.5)

        fc = np.array(fc)
        bbx = np.array(bbx)

        yield img_filename, img.shape, fc, bbx


def generate_onet_dataset():
    labels = load_labels(os.path.join(WIDER_TRAIN_DIR, "labels.json"))
    onet_labels = []
    count = dict(pos=0, neg=0, partial=0)
    c = 0

    for img_filename, img_shape, fc_pred, bbx_pred in compute_rnet_preds():
        c += 1
        for box in bbx_pred:
            mid_x = (2*box[0] + box[2])/2
            mid_y = (2*box[1] + box[3])/2
            hw = min(img_shape[0]-2, img_shape[1]-2, max(1, max(box[2], box[3])*1.1))
            x0 = int(min(img_shape[0]-hw-1, max(0, int(mid_x - hw/2))))
            y0 = int(min(img_shape[1]-hw-1, max(0, int(mid_y - hw/2))))
            hw = int(hw)
            box = (x0, y0, hw, hw)
            kind, mask_fc, mask_bbx, box_fc, box_label = get_box_label(box, labels[img_filename]["bbx"])

            if kind == "partial" and count[kind] <= min(list(count.values())) + 3:
                count[kind] += 1
                onet_labels.append((img_filename, mask_fc, mask_bbx, box_fc, box_label, box))
            elif count[kind] <= 3*min(list(count.values())) + 3:
                count[kind] += 1
                onet_labels.append((img_filename, mask_fc, mask_bbx, box_fc, box_label, box))

        if c % 100 == 0:
            print(c, count)
            with open(os.path.join(WIDER_TRAIN_DIR, f"onet_labels.json"), "w") as pnet_labels_file:
                json.dump(onet_labels, pnet_labels_file)


def generate_celeba_sample(img_shape, box, fll):
    size = max(box[2], box[3])
    size = int(np.random.uniform(0.8, 1.3)*size)
    size = max(12, size)
    random_box = np.empty((4,), dtype=np.int32)

    low = max(0, box[0]-int(size*0.3))
    high = min(box[0]+int(size*0.3), img_shape[0]-size)
    if low >= high:
        return None
    random_box[0] = np.random.randint(low, high, dtype=int)

    low = max(0, box[1]-int(size*0.3))
    high = min(box[1]+int(size*0.3), img_shape[1]-size)
    if low >= high:
        return None
    random_box[1] = np.random.randint(low, high, dtype=int)

    random_box[2] = size
    random_box[3] = size
    if iou_xyhw(box, random_box) > 0.65:
        fc_mask = True
        box_mask = True
        box_label = ((box[0]-random_box[0])/size, (box[1]-random_box[1])/size, box[2]/size, box[3]/size)
        fll_label = tuple([(fll[i] - random_box[i%2])/size for i in range(10)])
    elif iou_xyhw(box, random_box) > 0.3:
        fc_mask = False
        box_mask = True
        box_label = ((box[0]-random_box[0])/size, (box[1]-random_box[1])/size, box[2]/size, box[3]/size)
        fll_label = tuple([(fll[i] - random_box[i%2])/size for i in range(10)])
    else:
        return True, False, (0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    return fc_mask, box_mask, box_label, fll_label


def add_fll_labels(model):
    celeba_labels_path = os.path.join(CELEBA_DIR, "labels.json")
    labels_path = os.path.join(WIDER_TRAIN_DIR, model+"_labels.json")
    labels = load_labels(labels_path)
    celeba_labels = load_labels(celeba_labels_path)
    new_labels = []
    nb_pos = 0

    for img_filename, mask_fc, mask_bbx, box_fc, box_label, box_cut in tqdm(labels, desc="Adding fll mask"):
        new_labels.append((img_filename, mask_fc, mask_bbx, False, box_fc, box_label, (0, 0, 0, 0, 0, 0, 0, 0, 0, 0), box_cut))
        nb_pos += mask_fc * mask_bbx

    celeba_img_filenames = list(celeba_labels.keys())
    celeba_img_filenames = rd.sample(celeba_img_filenames, nb_pos)
    for img_filename in tqdm(celeba_img_filenames, desc="Adding celeba images"):
        img_shape = np.array(Image.open(os.path.join(CELEBA_DIR, "img_celeba", img_filename))).shape
        bbx, fll = celeba_labels[img_filename]["box"], celeba_labels[img_filename]["fll"]
        res = generate_celeba_sample(img_shape, bbx, fll)
        if res is not None:
            fc_mask, box_mask, box_label, fll_label = res
            new_labels.append((img_filename, fc_mask, False, True, (1, 0), box_label, fll_label, bbx))

    with open(os.path.join(CELEBA_DIR, f"{model}_labels.json"), "w") as labels_file:
        json.dump(new_labels, labels_file)
