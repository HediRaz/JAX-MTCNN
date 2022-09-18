from typing import Tuple, Dict, List
from functools import partial

import os
import jax
import jax.numpy as jnp
import numpy as np
import cv2
import haiku as hk
import pickle

from tqdm import tqdm

from mtcnn.utils import iou_xyhw
from mtcnn.networks import pnet_encoding_t, pnet_bbx_t, pnet_fc_t
from mtcnn.networks import rnet_encoding_t, rnet_fc_t, rnet_bbx_t, rnet_fll_t
from mtcnn.networks import onet_encoding_t, onet_fc_t, onet_bbx_t, onet_fll_t


@jax.jit
def floor_2powlog2(n):
    return jnp.int32(jnp.exp2(jnp.int32(jnp.log2(n))))


class Pyramid:
    def __init__(self, scale_factor: float = 0.7):
        self.scale_factor = scale_factor

    def body_scale_factors(i, val):
        factors, current_factor, scale_factor = val
        current_factor *= scale_factor
        factors = factors.at[i].set(current_factor)
        return factors, current_factor, scale_factor

    @partial(jax.jit, static_argnums=(0, 1))
    def compute_scale_factors(scale_factor, img_shape) -> List[float]:
        mini_hw = min(img_shape[:2])
        p = int(np.log(12 / mini_hw) / np.log(scale_factor))
        factors = jnp.ones((p,), dtype=jnp.float32)
        factors, _, _ = jax.lax.fori_loop(1, p, Pyramid.body_scale_factors, (factors, 1., scale_factor))
        return factors

    @partial(jax.jit, static_argnums=(0, 1))
    def resize_image(scale_factor: float, img_shape, img: jnp.ndarray) -> jnp.ndarray:
        new_shape = [int(img_shape[0]*scale_factor),
                     int(img_shape[1]*scale_factor), img_shape[2]]
        img = jax.image.resize(img, new_shape, "bicubic")
        return img


class SoftNMS:
    """ Soft Non Maximum Suppression """
    def __init__(self, sigma: float = 100):
        self.sigma = sigma
        self.possible_length = [2**i for i in range(10)]

    @partial(jax.jit, static_argnums=0)
    def iou_matrix(n: int, bbx: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(jax.vmap(iou_xyhw, in_axes=(0, None)), in_axes=(None, 0))(bbx, bbx)

    @partial(jax.jit, static_argnums=(0, 1))
    def call_1(n: int, sigma: float, fc: jnp.ndarray, bbx: jnp.ndarray, threshold: jnp.ndarray):
        indexes = jnp.argsort(-fc.at[:, 0].get())
        fc = fc.at[indexes].get()
        bbx = bbx.at[indexes].get()

        iou_m = SoftNMS.iou_matrix(n, bbx)
        iou_m = jnp.triu(iou_m, k=1)
        iou_m = -jnp.square(iou_m)/sigma

        fc = fc.at[:, 0].set(jnp.exp(jnp.log(fc.at[:, 0].get()) + jnp.sum(iou_m, 0)))
        fc = jax.nn.softmax(fc)
        new_n = floor_2powlog2(jnp.sum(jnp.where(fc.at[:, 0].get() > threshold, 1, 0)))
        # new_n = 32
        return fc, bbx, new_n

    @partial(jax.jit, static_argnums=(0, 1))
    def call_2(n: int, new_n: int, fc: jnp.ndarray, bbx: jnp.ndarray, fll: jnp.ndarray):
        indexes = jnp.argsort(-fc.at[:, 0].get()).at[:new_n].get()
        return fc.at[indexes].get(), bbx.at[indexes].get(), fll.at[indexes].get()

    def __call__(self, fc: jnp.ndarray, bbx: jnp.ndarray, fll: jnp.ndarray, threshold: float):
        """
        Compute new scores of predictions according to SoftNMS algorithm

        Args:
          fc: face classification (shape (n, 2))
          bbx: bounding boxes (shape (n, 4))
          fll: face landmarks localisation (shape (n, 10))
          threshold: score threshold to prune bbx

        Returns:
          new scores, bbx and fll

        """
        n = len(fc)
        if n == 0:
            return fc, bbx, fll
        fc, bbx, new_n = SoftNMS.call_1(n, self.sigma, fc, bbx, threshold)
        fc, bbx, fll = SoftNMS.call_2(n, int(new_n), fc, bbx, fll)
        return fc, bbx, fll


class NMS_false:
    @staticmethod
    def __call_body_fn_body_fn(i, val):
        bbx, threshold, sorted_idx, keep_idx, bbx_m = val
        n = sorted_idx.at[i].get()
        keep_idx = keep_idx.at[n].set((iou_xyhw(bbx_m, bbx.at[n].get()) < threshold) & keep_idx.at[n].get())
        return bbx, threshold, sorted_idx, keep_idx, bbx_m

    @staticmethod
    def __call_body_fn(i, val):
        bbx, threshold, sorted_idx, keep_idx = val
        m = sorted_idx.at[i].get()
        bbx_m = bbx.at[m].get()
        (bbx, threshold, sorted_idx, keep_idx, bbx_m) = jax.lax.fori_loop(
            i+1,
            len(sorted_idx),
            NMS_false.__call_body_fn_body_fn,
            (bbx, threshold, sorted_idx, keep_idx, bbx_m))
        return bbx, threshold, sorted_idx, keep_idx

    def __call__(self, fc: jnp.ndarray, bbx: jnp.ndarray, iou_threshold: float):
        """
        Args:
          fc: 1D-Array with prediciton scores for each bbx
          bbx: 2D-Array with boxes [x, y, h, w]
          iou_threshold: threshold for pruning boxes

        Return:
          new scores, new boxes

        """
        sorted_idx = jnp.argsort(-fc.at[:, 0].get())
        keep_idx = jnp.ones(sorted_idx.shape, dtype=jnp.bool_)
        for i in range(len(sorted_idx)):
            bbx, iou_threshold, sorted_idx, keep_idx = NMS_false.__call_body_fn(i, (bbx, iou_threshold, sorted_idx, keep_idx))
        # bbx, iou_threshold, sorted_idx, keep_idx = jax.lax.fori_loop(0, len(sorted_idx), NMS.__call_body_fn, (bbx, iou_threshold, sorted_idx, keep_idx))
        return fc.at[keep_idx].get(), bbx.at[keep_idx].get()


class MTCNN:
    def __init__(self, key=jax.random.PRNGKey(42)):
        self.img_shape = (1024, 1024)
        self.idx_mask_x = jnp.transpose(jnp.tile(jnp.arange(self.img_shape[0]), (self.img_shape[1], 1)))  # jnp.where(jnp.ones((self.img_shape)))[0]
        self.idx_mask_y = jnp.tile(jnp.arange(self.img_shape[1]), (self.img_shape[0], 1))   # jnp.where(jnp.ones((self.img_shape)))[1]

        self.pyramid = Pyramid()
        self.nms = SoftNMS(sigma=0.5)

        self.pnet_threshold = 0.7
        self.rnet_threshold = 0.4
        self.onet_threshold = 0.6

        self.load_model("pnet", "models/pnet/params19")
        self.load_model("rnet", "models/rnet/params20")
        self.load_model("onet", "models/onet/params18")

    def load_model(self, model, path):
        """
        Load model from path

        Args:
        model: "pnet", "rnet" or "onet"
        path: path to weights

        """
        if model not in ["pnet", "rnet", "onet"]:
            raise ValueError("argument 'model' should be one of 'pnet', 'rnet', or 'onet")
        if not os.path.exists(path):
            raise FileExistsError(f"path {path} does not exist")

        with open(path, "rb") as params_file:
            if model == "pnet":
                self.pnet_params = pickle.load(params_file)
            if model == "rnet":
                self.rnet_params = pickle.load(params_file)
            if model == "onet":
                self.onet_params = pickle.load(params_file)

    def precompile(self, img_shape):
        key = jax.random.PRNGKey(42)
        img = jax.random.uniform(key, img_shape)

        print("Precompiling PNet")
        MTCNN.process_img(img_shape, img)
        fc_pnet, bbx_pnet = MTCNN.pnet_inference_1(img_shape, self.pnet_params, img)
        MTCNN.get_preds_pnet_1(img_shape, self.pnet_threshold, fc_pnet)
        for new_n in tqdm([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]):
            fc, bbx = MTCNN.get_preds_pnet_2(img_shape, new_n, fc_pnet, bbx_pnet, self.idx_mask_x, self.idx_mask_y, 1.)
            # MTCNN.pnet_prune(new_n, fc, bbx)

        print("Precompiling RNet")
        for n in tqdm([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]):
            bbx_pnet = jax.random.uniform(key, (n, 4))
            x0, y0, hw = MTCNN.rnet_inference_1(img_shape, n, bbx_pnet)
            img_bbx = MTCNN.rnet_inference_2(np.array(img), x0, y0, hw)
            fc_rnet, bbx_rnet, fll_rnet, _ = MTCNN.rnet_inference_3(n, self.rnet_params, img_bbx, self.rnet_threshold)
            for new_n in tqdm([1, 2, 4, 8, 16, 32, 64, 128, 256]):
                fc_n, old_bbx_n, new_bbx_n, fll_n, hw_n = MTCNN.rnet_inference_4(n, new_n, fc_rnet, bbx_pnet, bbx_rnet, fll_rnet, hw)
                MTCNN.rnet_inference_5(new_n, old_bbx_n, new_bbx_n, fll_n, hw_n)

        print("Precompiling ONet")
        for n in tqdm([1, 2, 4, 8, 16, 32, 64, 128, 256]):
            bbx_pnet = jax.random.uniform(key, (n, 4))
            x0, y0, hw = MTCNN.rnet_inference_1(img_shape, n, bbx_pnet)
            img_bbx = MTCNN.onet_inference_2(np.array(img), x0, y0, hw)
            fc_rnet, bbx_rnet, fll_rnet, _ = MTCNN.onet_inference_3(n, self.onet_params, img_bbx, self.onet_threshold)

    # @partial(jax.jit, static_argnums=0)
    def process_img(img_shape, img):
        return (img / 255.)

    @partial(jax.jit, static_argnums=0)
    def get_preds_pnet_1(img_shape: Tuple[int], pnet_threshold: float, fc_pred: jnp.ndarray):
        # Number of positive predictions
        n = jnp.sum(jnp.where(fc_pred.at[:, :, 0].get() > pnet_threshold, 1, 0))
        # Average to the next power of 2 for compilation issue
        new_n = floor_2powlog2(n)
        return fc_pred, new_n

    @partial(jax.jit, static_argnums=(0, 1))
    def get_preds_pnet_2(img_shape: Tuple[int], new_n: int,
                         fc_pred: jnp.ndarray, bbx_pred: jnp.ndarray, x_mask: jnp.ndarray, y_mask: jnp.ndarray, scale_factor: float):
        # Reshape preds
        res_shape = (img_shape[0] - 2) // 2 - 4, (img_shape[1] - 2) // 2 - 4
        fc_pred = fc_pred.reshape((res_shape[0] * res_shape[1], 2))
        bbx_pred = bbx_pred.reshape((res_shape[0] * res_shape[1], 4))
        x_mask = x_mask.at[:res_shape[0], :res_shape[1]].get().reshape((res_shape[0] * res_shape[1]))
        y_mask = y_mask.at[:res_shape[0], :res_shape[1]].get().reshape((res_shape[0] * res_shape[1]))
        # Sort predictions
        indexes = jnp.argsort(-fc_pred.at[:, 0].get()).at[:new_n].get()
        # get nex_n best preds
        fc_pred = fc_pred.at[indexes].get(indices_are_sorted=True, unique_indices=True)
        bbx_pred = bbx_pred.at[indexes].get(indices_are_sorted=True, unique_indices=True)
        x_mask = x_mask.at[indexes].get(indices_are_sorted=True, unique_indices=True)
        y_mask = y_mask.at[indexes].get(indices_are_sorted=True, unique_indices=True)
        # Compute real bbx
        bbx_pred *= 12
        bbx_pred = bbx_pred.at[:, 0].add(x_mask*2)
        bbx_pred = bbx_pred.at[:, 1].add(y_mask*2)
        bbx_pred = jnp.int32(jnp.rint(bbx_pred/scale_factor))
        return fc_pred, bbx_pred

    def get_preds_pnet(self, fc_pred: jnp.ndarray, bbx_pred: jnp.ndarray, scale_factor: float, img_shape: Tuple[int]):
        """ Process PNet results """
        fc_pred, new_n = MTCNN.get_preds_pnet_1(
            img_shape,
            self.pnet_threshold,
            fc_pred
        )
        return MTCNN.get_preds_pnet_2(img_shape, int(new_n), fc_pred, bbx_pred, self.idx_mask_x, self.idx_mask_y, scale_factor)

    @partial(jax.jit, static_argnums=0)
    def pnet_inference_1(img_shape: tuple, pnet_params: Dict[str, hk.Params], img: jnp.ndarray):
        img = pnet_encoding_t.apply(pnet_params["encoding"], jnp.expand_dims(img, 0))
        fc_pred = pnet_fc_t.apply(pnet_params["fc"], img)[0]
        bbx_pred = pnet_bbx_t.apply(pnet_params["bbx"], img)[0]
        return fc_pred, bbx_pred

    def pnet_inference(self, img: jnp.ndarray, scale_factor: float):
        """ Apply PNet to img and process results """
        fc_pred, bbx_pred = MTCNN.pnet_inference_1(img.shape, self.pnet_params, img)
        fc_pred, bbx_pred = self.get_preds_pnet(fc_pred, bbx_pred, scale_factor, img.shape)
        return fc_pred, bbx_pred

    @partial(jax.jit, static_argnums=(0, 1))
    def pnet_prune(n, new_n, fc, bbx):
        indexes = jnp.argsort(-fc.at[:, 0].get()).at[:new_n].get()
        # get nex_n best preds
        fc = fc.at[indexes].get(indices_are_sorted=True, unique_indices=True)
        bbx = bbx.at[indexes].get(indices_are_sorted=True, unique_indices=True)
        return fc, bbx

    @partial(jax.vmap, in_axes=(0, None, None))
    def compute_cut_box(box: jnp.array, img_shape: tuple, zoom_factor: float = 1.) -> Tuple[int]:
        """ Compute square box to cut """
        mid_x = (2*box.at[0].get() + box.at[2].get())/2
        mid_y = (2*box.at[1].get() + box.at[3].get())/2
        hw = jnp.maximum(1, jnp.maximum(box.at[2].get(), box.at[3].get())*zoom_factor)
        x0 = jnp.int32(jnp.minimum(img_shape[0]-hw-1, jnp.maximum(0, jnp.fix(mid_x - hw/2))))
        y0 = jnp.int32(jnp.minimum(img_shape[1]-hw-1, jnp.maximum(0, jnp.fix(mid_y - hw/2))))
        hw = jnp.int32(hw)
        return x0, y0, hw

    def rnet_cut_image(img: np.ndarray, x0: np.ndarray, y0: np.ndarray, hw: np.ndarray):
        """ Crop box from image """
        # box_img = jax.lax.dynamic_slice(img, (x0, y0, 0), (hw, hw, 3))
        # box_img = MTCNN.rnet_resize(box_img)
        # print(x0.shape)
        box_img = img[x0: x0+hw, y0: y0+hw]
        box_img = cv2.resize(box_img, (24, 24), interpolation=cv2.INTER_CUBIC)
        return box_img

    def rnet_resize(img):
        return jax.image.resize(img, (24, 24, 3), "bicubic")

    @partial(jax.jit, static_argnums=(0, 1))
    def rnet_inference_1(img_shape: tuple, n: int, bbx: jnp.ndarray):
        # Cut and resize box from img
        x0, y0, hw = MTCNN.compute_cut_box(bbx, img_shape, 1.1)
        return x0, y0, hw

    def rnet_inference_2(img_numpy: np.ndarray, x0: jnp.ndarray, y0: jnp.ndarray, hw: jnp.ndarray):
        x0 = np.array(x0)
        y0 = np.array(y0)
        hw = np.array(hw)
        img_bbx = jnp.array(list(map(lambda x, y, hw: MTCNN.rnet_cut_image(img_numpy, x, y, hw), x0, y0, hw)))
        return img_bbx

    @partial(jax.jit, static_argnums=0)
    def rnet_inference_3(n: int, rnet_params: Dict[str, hk.Params], img_bbx: jnp.ndarray, rnet_threshold: float):
        # Apply net
        encoding_bbx = rnet_encoding_t.apply(rnet_params["encoding"], img_bbx)
        fc_pred = rnet_fc_t.apply(rnet_params["fc"], encoding_bbx)
        new_bbx = rnet_bbx_t.apply(rnet_params["bbx"], encoding_bbx)
        fll_pred = rnet_fll_t.apply(rnet_params["fll"], encoding_bbx)
        n = jnp.sum(jnp.where(fc_pred.at[:, 0].get() > rnet_threshold, 1, 0))
        new_n = floor_2powlog2(n)
        return fc_pred, new_bbx, fll_pred, new_n

    @partial(jax.jit, static_argnums=(0, 1))
    def rnet_inference_4(n: int, new_n: int, fc_pred: jnp.ndarray, old_bbx: jnp.ndarray, new_bbx: jnp.ndarray, fll_pred: jnp.ndarray, hw: jnp.ndarray):
        # Get best preds only
        indexes = jnp.argsort(- fc_pred.at[:, 0].get()).at[:new_n].get()
        fc_pred = fc_pred.at[indexes].get(indices_are_sorted=True, unique_indices=True)
        old_bbx = old_bbx.at[indexes].get(indices_are_sorted=True, unique_indices=True)
        new_bbx = new_bbx.at[indexes].get(indices_are_sorted=True, unique_indices=True)
        fll_pred = fll_pred.at[indexes].get(indices_are_sorted=True, unique_indices=True)
        hw = hw.at[indexes].get(indices_are_sorted=True, unique_indices=True)
        return fc_pred, old_bbx, new_bbx, fll_pred, hw

    @partial(jax.jit, static_argnums=0)
    def rnet_inference_5(n: int, old_bbx: jnp.ndarray, new_bbx: jnp.ndarray, fll_pred: jnp.ndarray, hw: jnp.ndarray):
        # Compute predicted box
        new_bbx = new_bbx.at[:, 2:].multiply(jnp.tile(hw, (2, 1)).transpose())
        new_bbx = new_bbx.at[:, :2].add(old_bbx.at[:, :2].get())
        new_bbx = jnp.int32(jnp.rint(new_bbx))

        # Compute predicted landmarks
        fll_pred = jax.vmap(jnp.dot, in_axes=(0, 0))(fll_pred, hw)
        old_bbx_x = old_bbx.at[:, 0].get()
        old_bbx_y = old_bbx.at[:, 1].get()
        fll_pred = fll_pred.at[:, ::2].add(jnp.tile(old_bbx_x, (5, 1)).transpose())
        fll_pred = fll_pred.at[:, 1::2].add(jnp.tile(old_bbx_y, (5, 1)).transpose())
        fll_pred = jnp.int32(jnp.rint(fll_pred))

        return new_bbx, fll_pred

    def rnet_inference(self, img_numpy: np.array, img: jnp.ndarray, bbx_pred: jnp.ndarray):
        """ Apply RNet to img on predicted bbx """
        n = len(bbx_pred)
        img_shape = (int(img.shape[0]), int(img.shape[1]), 3)
        x0, y0, hw = MTCNN.rnet_inference_1(img_shape, len(bbx_pred), bbx_pred)
        img_bbx = MTCNN.rnet_inference_2(img_numpy, x0, y0, hw)
        fc_pred, new_bbx, fll_pred, new_n = MTCNN.rnet_inference_3(n, self.rnet_params, img_bbx, self.rnet_threshold)
        new_n = int(new_n)
        fc_pred, old_bbx, new_bbx, fll_pred, hw = MTCNN.rnet_inference_4(n, new_n, fc_pred, bbx_pred, new_bbx, fll_pred, hw)
        new_bbx, fll_pred = MTCNN.rnet_inference_5(new_n, old_bbx, new_bbx, fll_pred, hw)
        return fc_pred, new_bbx, fll_pred

    def onet_cut_image(img, x0, y0, hw):
        """ Crop box from image """
        box_img = img[x0: x0+hw, y0: y0+hw]
        box_img = cv2.resize(box_img, (48, 48), interpolation=cv2.INTER_CUBIC)
        return box_img

    def onet_inference_2(img_numpy, x0, y0, hw):
        x0 = np.array(x0)
        y0 = np.array(y0)
        hw = np.array(hw)
        img_bbx = jnp.array(list(map(lambda x, y, hw: MTCNN.onet_cut_image(img_numpy, x, y, hw), x0, y0, hw)))
        return img_bbx

    @partial(jax.jit, static_argnums=0)
    def onet_inference_3(n, onet_params: Dict[str, hk.Params], img_bbx, onet_threshold):
        # Apply net
        encoding_bbx = jax.jit(onet_encoding_t.apply)(onet_params["encoding"], img_bbx)
        fc_pred = jax.jit(onet_fc_t.apply)(onet_params["fc"], encoding_bbx)
        new_bbx = jax.jit(onet_bbx_t.apply)(onet_params["bbx"], encoding_bbx)
        fll_pred = jax.jit(onet_fll_t.apply)(onet_params["fll"], encoding_bbx)
        n = jnp.sum(jnp.where(fc_pred > onet_threshold, 1, 0))
        new_n = floor_2powlog2(n)
        return fc_pred, new_bbx, fll_pred, new_n

    def onet_inference(self, img_numpy, img, bbx_pred):
        n = len(bbx_pred)
        img_shape = (int(img.shape[0]), int(img.shape[1]), 3)
        x0, y0, hw = MTCNN.rnet_inference_1(img_shape, len(bbx_pred), bbx_pred)
        img_bbx = MTCNN.onet_inference_2(img_numpy, x0, y0, hw)
        fc_pred, new_bbx, fll_pred, new_n = MTCNN.onet_inference_3(n, self.onet_params, img_bbx, self.onet_threshold)
        # new_n = int(new_n)
        new_n = n
        old_bbx = bbx_pred
        # fc_pred, old_bbx, new_bbx, fll_pred, hw = MTCNN.rnet_inference_4(n, new_n, fc_pred, bbx_pred, new_bbx, fll_pred, hw)
        new_bbx, fll_pred = MTCNN.rnet_inference_5(new_n, old_bbx, new_bbx, fll_pred, hw)
        return fc_pred, new_bbx, fll_pred

    def __call__(self, img_numpy: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Apply MTCNN on img

        Args:
          img: Image to process not processed

        Returns:
          fc, bbx, fll: scores, bbx and fll predicted on img
        """
        img_numpy = MTCNN.process_img(img_numpy.shape, img_numpy)
        img = jnp.array(img_numpy)
        p_factors = Pyramid.compute_scale_factors(self.pyramid.scale_factor, img.shape)
        fc, bbx = [], []
        for factor in np.array(p_factors):
            resized_img = Pyramid.resize_image(factor, img.shape, img)
            fc_i, bbx_i = self.pnet_inference(resized_img, factor)
            fc.append(fc_i)
            bbx.append(bbx_i)
        del fc_i
        del bbx_i
        fc = jnp.concatenate(fc, 0)
        bbx = jnp.concatenate(bbx, 0)
        if len(fc) == 0:
            return fc, bbx, None
        new_n = int(floor_2powlog2(fc.shape[0]))
        fc, bbx = MTCNN.pnet_prune(len(fc), new_n, fc, bbx)

        fc, bbx, _ = self.nms(fc, bbx, bbx, 0.5)
        if len(fc) == 0:
            return fc, bbx, None

        fc, bbx, _ = self.rnet_inference(img_numpy, img, bbx)
        fc, bbx, _ = self.nms(fc, bbx, _, 0.4)
        if len(fc) == 0:
            return fc, bbx, _

        fc, bbx, fll = self.onet_inference(img_numpy, img, bbx)
        fc, bbx, _ = SoftNMS.call_1(len(fc), 0.3, fc, bbx, 0.6)  # self.nms(fc, bbx, fll, 0.6)
        new_n = jnp.sum(jnp.where(fc.at[:, 0].get() > 0.6, 1, 0))
        fc = fc.at[:new_n].get()
        bbx = bbx.at[:new_n].get()
        fll = fll.at[:new_n].get()
        return fc, bbx, fll
