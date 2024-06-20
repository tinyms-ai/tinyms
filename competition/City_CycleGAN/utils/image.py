import math

import numpy as np
from PIL import Image


def save_image(img, filename):
    img = ((img + 1) / 2 * 255).astype(np.uint8)
    img_pil = Image.fromarray(img)
    img_pil.save(filename)


def concatenate_images(tensor_images, cols=None):
    if cols is None:
        cols = math.ceil(math.sqrt(len(tensor_images)))
    images = []
    for image in tensor_images:
        images.append(image.asnumpy())
    rows = []
    for i in range(math.ceil(len(images) / cols)):
        row_image = np.concatenate(images[i * cols:(i + 1) * cols], axis=2)
        rows.append(row_image)
    res = np.concatenate(rows, axis=3)
    res = res.transpose((0, 3, 2, 1))
    res = np.concatenate(res, axis=0)
    res = res.transpose((1, 0, 2))
    return res
