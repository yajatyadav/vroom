import logging
import numpy as np
import cv2


logger = logging.getLogger("io")


def save_array(file, array):
    array = array.transpose(1, 2, 0)
    array_out = np.uint16(array[...,::-1]*65535.)
    cv2.imwrite(str(file), array_out)


def save_image(file, img):
    img = img.transpose(1, 2, 0)
    im_out = np.uint8(img[...,::-1]*255.)
    cv2.imwrite(str(file), im_out)


def save_npy(file, array):
    np.save(str(file), array)


def load_array(file):
    array = cv2.imread(str(file), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if len(array.shape) == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    return np.array(array, dtype=np.float32) / 65535.


def load_image(file):
    array = cv2.cvtColor(cv2.imread(str(file), -1), cv2.COLOR_BGR2RGB)
    max_v = 65535. if array.dtype == np.uint16 else 255.
    array = np.array(array, dtype=np.float32) / max_v
    return array


def save_flow(file, flow, scaling=100):
    flow = flow.transpose(1, 2, 0).astype(np.float64)
    min_value = np.iinfo(np.uint16).min
    max_value = np.iinfo(np.uint16).max
    flow = flow * scaling + float(max_value) / 2

    flow = np.round(flow)
    
    if np.any(flow < min_value) or np.any(flow > max_value):
        logger.warning(f"Flow values out of bounds: {flow.min()} - {flow.max()}")

    flow = np.clip(flow, min_value, max_value).astype(np.uint16)

    flow = np.concatenate([flow, np.zeros_like(flow[..., :1])], axis=-1)

    cv2.imwrite(str(file), flow)


def load_flow(file, scaling=100):
    flow = cv2.imread(str(file), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    min_value = np.iinfo(np.uint16).min
    max_value = np.iinfo(np.uint16).max
    flow = np.array(flow, dtype=np.float64)[..., :2]
    flow = (flow - float(max_value) / 2) / scaling
    flow = np.transpose(flow, (2, 0, 1)).astype(np.float32)
    return flow


def load_npy(file):
    return np.load(file)