'''Util module to log images'''
from os import mkdir, path
from PIL.Image import Image
import cv2
from afy.custom_typings import CV2Image

class ImageLogger():
    '''Class that logs the images in a certain folder'''
    counter = 0

    def __init__(self, save_path: str):
        self.path = save_path
        if not path.exists(save_path):
            mkdir(save_path)

    @property
    def filename(self):
        filename = f'{self.counter:08}'
        self.counter += 1
        return filename

    @property
    def full_path(self):
        return path.join(self.path, self.filename)

    def save_pil(self, image: Image):
        image.save(self.full_path)

    def save_cv2(self, image: CV2Image):
        cv2.imwrite(self.full_path, image)
