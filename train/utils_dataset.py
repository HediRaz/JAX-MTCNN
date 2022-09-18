import os
import json
import numpy as np
import cv2
from PIL import Image
from torch.nn import Sequential
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T



WIDER_TRAIN_DIR = "datasets/WIDER/WIDER_train"
CELEBA_DIR = "datasets/CELEBA"


TRAIN_TRANSFORMS = Sequential(
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2),
)


def load_labels(path):
    with open(path, 'r') as file:
        labels = json.load(file)
    return labels


def iou_xyhw(bbx1: np.ndarray, bbx2: np.ndarray) -> float:
    """ Compute IOU of two bbx [x, y, h, w] """
    h = np.maximum(0, np.minimum(bbx1[0]+bbx1[2], bbx2[0]+bbx2[2]) - np.maximum(bbx1[0], bbx2[0]))
    w = np.maximum(0, np.minimum(bbx1[1]+bbx1[3], bbx2[1]+bbx2[3]) - np.maximum(bbx1[1], bbx2[1]))
    a1 = bbx1[2] * bbx1[3]
    a2 = bbx2[2] * bbx2[3]
    inter = h*w
    return float(inter / (a1 + a2 - inter))


class PNetDataset(Dataset):
    def __init__(self, model):
        super().__init__()
        with open(os.path.join(WIDER_TRAIN_DIR, f"{model}_labels.json"), "r") as labels_file:
            labels = json.load(labels_file)
        pos_samples = labels["pos"]
        neg_samples = labels["neg"][:2*len(pos_samples)]
        self.data = pos_samples
        self.data.extend(neg_samples)

        self.resize_dim = (12, 12)
        self.train = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_filename, fc, bbx, bbx_cut = self.data[idx]
        img = Image.open(os.path.join(WIDER_TRAIN_DIR, "images", img_filename))
        if self.train:
            img = TRAIN_TRANSFORMS(img)
        img = np.array(img)
        img = img[bbx_cut[0]: bbx_cut[0]+bbx_cut[2], bbx_cut[1]: bbx_cut[1]+bbx_cut[3]]
        img = cv2.resize(img, self.resize_dim, interpolation=cv2.INTER_CUBIC)
        img = np.array(img) / 255
        fc = np.array(fc, dtype=np.float32)
        bbx = np.array(bbx, dtype=np.float32)
        return img, fc, bbx


class RONetDataset(Dataset):
    def __init__(self, model):
        super().__init__()
        with open(os.path.join(CELEBA_DIR, f"{model}_labels.json"), "r") as labels_file:
            labels = json.load(labels_file)
        self.data = labels

        if model == "rnet":
            self.resize_dim = (24, 24)
        elif model == "onet":
            self.resize_dim = (48, 48)

        self.train = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_filename, mask_fc, mask_bbx, mask_fll, fc_label, bbx_label, fll_label, bbx_cut = self.data[idx]
        if mask_fll:
            img = Image.open(os.path.join(CELEBA_DIR, "img_celeba", img_filename))
        else:
            img = Image.open(os.path.join(WIDER_TRAIN_DIR, "images", img_filename))
        if self.train:
            img = TRAIN_TRANSFORMS(img)
        img = np.array(img)
        img = img[bbx_cut[0]: bbx_cut[0]+bbx_cut[2], bbx_cut[1]: bbx_cut[1]+bbx_cut[3]]
        img = cv2.resize(img, self.resize_dim, interpolation=cv2.INTER_CUBIC)
        img = np.array(img) / 255
        fc_label = np.array(fc_label, dtype=np.float32)
        bbx_label = np.array(bbx_label, dtype=np.float32)
        fll_label = np.array(fll_label)
        return img, mask_fc, mask_bbx, mask_fll, fc_label, bbx_label, fll_label


class RNetDataset(RONetDataset):
    def __init__(self):
        super().__init__("rnet")


class ONetDataset(RONetDataset):
    def __init__(self):
        super().__init__("onet")


def pnet_collate_fn(batch):
    batch_img = np.stack([np.array(item[0]) for item in batch], 0)
    batch_fc = np.stack([np.array(item[1]) for item in batch], 0)
    batch_bbx = np.stack([np.array(item[2]) for item in batch], 0)
    batch = {"img": batch_img, "fc": batch_fc, "bbx": batch_bbx}
    return batch


def ronet_collate_fn(batch):
    batch_img = np.stack([np.array(item[0]) for item in batch], 0)
    batch_mask_fc = np.stack([np.array(item[1]) for item in batch], 0)
    batch_mask_bbx = np.stack([np.array(item[2]) for item in batch], 0)
    batch_mask_fll = np.stack([np.array(item[3]) for item in batch], 0)
    batch_fc = np.stack([np.array(item[4]) for item in batch], 0)
    batch_bbx = np.stack([np.array(item[5]) for item in batch], 0)
    batch_fll = np.stack([np.array(item[6]) for item in batch], 0)
    batch = {"img": batch_img, "mask_fc": batch_mask_fc, "mask_bbx": batch_mask_bbx, "mask_fll": batch_mask_fll, "fc": batch_fc, "bbx": batch_bbx, "fll": batch_fll}
    return batch


def create_dataloaders(ds, batch_size=16, split=0.8):
    if isinstance(ds, PNetDataset):
        collate_fn = pnet_collate_fn
    elif isinstance(ds, RONetDataset):
        collate_fn = ronet_collate_fn
    n = len(ds)
    n_train = int(split * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(ds, [n_train, n - n_train])
    train_ds.dataset.train = True

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)

    return train_dl, val_dl
