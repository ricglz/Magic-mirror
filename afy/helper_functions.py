from argparse import ArgumentParser
from glob import glob

import cv2
import numpy as np

from afy.utils import resize

def prepare_image(img: np.ndarray, IMG_SIZE=512):
    if img.ndim == 2:
        img = np.tile(img[..., None], [1, 1, 3])
    img = img[..., :3][..., ::-1]
    img = resize(img, (IMG_SIZE, IMG_SIZE))
    return img

def load_images(opt: ArgumentParser, IMG_SIZE=512):
    avatars = []
    filenames = []
    images_list = sorted(glob(f'{opt.avatars}/*'))
    for i, f in enumerate(images_list):
        if f.endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(f)
            if img is None:
                continue

            img = prepare_image(img, IMG_SIZE)
            avatars.append(img)
            filenames.append(f)
    return avatars, filenames
