import jax
import jax.numpy as jnp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio.v3 as iio
import imageio.v2 as iio2
from time import time

from mtcnn.utils import draw_box, draw_fll
from mtcnn.mtcnn import MTCNN, Pyramid
print("Available Devices: ", jax.devices())


def pnet_video(video_path="test.mp4"):
    mtcnn = MTCNN()

    processed_frames = []
    for c, frame in enumerate(iio.imiter(video_path)):
        print(c, end="\r")
        if c > 500:
            break
        if (c+1) % 50 == 0:
            iio.imwrite("test_pnet.mp4", np.stack(processed_frames, 0), fps=25)
        img = np.array(frame) / 255

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
        fc, bbx, _ = mtcnn.nms(fc, bbx, bbx, 0.2)
        bbx = np.array(bbx)

        for box in bbx:
            # frame = draw_box(frame, [box[1], box[0], box[2], box[3]])
            frame = draw_box(frame, box)

        processed_frames.append(frame)


def rnet_video(video_path="test.mp4"):
    mtcnn = MTCNN()
    mtcnn.pnet_threshold = 0.8

    processed_frames = []
    for c, frame in enumerate(iio.imiter(video_path)):
        print(c)
        if (c+1) % 50 == 0:
            iio.imwrite("test_rnet.mp4", np.stack(processed_frames, 0), fps=25)
        img = np.array(frame) / 255

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
        print("PNet", len(fc))
        fc, bbx, _ = mtcnn.nms(fc, bbx, bbx, 0.4)
        print("PNet after nms", len(fc))
        if len(fc) != 0:
            fc, bbx, _ = mtcnn.rnet_inference(img, bbx)
            fc, bbx, _ = mtcnn.nms(fc, bbx, bbx, 0.5)
        bbx = np.array(bbx)
        print("  RNet", len(fc))

        for box in bbx:
            # frame = draw_box(frame, [box[1], box[0], box[2], box[3]])
            frame = draw_box(frame, box)
        processed_frames.append(frame)

        # out.write(frame)


def mtcnn_video(video_path="test.mp4"):
    mtcnn = MTCNN()
    mtcnn.pnet_threshold = 0.8
    mtcnn.precompile((368, 640, 3))

    top = time()
    writer = iio2.get_writer("test_mtcnn.mp4", format="FFMPEG", mode="I", fps=25)
    for c, frame in enumerate(iio.imiter(video_path)):
        frame = cv2.resize(frame, (640, 368), interpolation=cv2.INTER_CUBIC)
        print(f"{int((c%50 +1)/(time()-top))} fps,  {c} frames processed  ", end="\r")
        if (c+1) % 50 == 0:
            # iio.imwrite("test_mtcnn.mp4", np.stack(processed_frames, 0), fps=25)
            top = time()
        img = np.array(frame)

        _, bbx, fll = mtcnn(img)

        for box, _fll in zip(bbx, fll):
            frame = draw_box(frame, box)
            frame = draw_fll(frame, _fll, (0, 255, 0))
        writer.append_data(frame)
    writer.close()


def mtcnn_cam():
    mtcnn = MTCNN()
    mtcnn.pnet_threshold = 0.8
    # mtcnn.precompile((368, 640, 3))
    plt.ion()

    top = time()
    reader = iio2.get_reader("<video0>")
    for c, frame in enumerate(reader):
        # frame = cv2.resize(frame, (640, 368), interpolation=cv2.INTER_CUBIC)
        print(f"{int((c%50 +1)/(time()-top))} fps,  {c} frames processed  ", end="\r")
        if (c+1) % 50 == 0:
            # iio.imwrite("test_mtcnn.mp4", np.stack(processed_frames, 0), fps=25)
            top = time()
        img = np.array(frame)

        _, bbx, fll = mtcnn(img)

        if bbx is not None and fll is not None and len(bbx) > 0:
            for box, _fll in zip(bbx, fll):
                frame = draw_box(frame, box)
                frame = draw_fll(frame, _fll, (0, 255, 0))
        
        # plt.imshow(frame)
        # plt.savefig("results/res_cam.png")
    reader.close()
