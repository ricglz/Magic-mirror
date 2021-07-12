from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch

from afy.custom_typings import CV2Image
from afy.magic_mirror import MagicMirror

class Predictor(ABC):
    '''Swapper abstract class'''
    output_size = (512, 512)

    def __init__(self, *_):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.magic_mirror = MagicMirror()

    @classmethod
    def __subclasshook__(cls, subclass):
        if hasattr(subclass, 'set_source_image') and callable(subclass.set_source_image) and \
           hasattr(subclass, '_predict') and callable(subclass._predict):
            return True
        return NotImplemented

    def reset_frames(self):
        pass

    @abstractmethod
    def set_source_image(self, source_image: CV2Image):
        raise NotImplementedError

    @abstractmethod
    def _predict(self, driving_frame: CV2Image):
        raise NotImplementedError

    def predict(self, driving_frame: CV2Image):
        if self.magic_mirror.should_predict():
            out = self._predict(driving_frame)
        else:
            out = driving_frame

        error_msg = f'Expected out to be np.ndarray, got {out.__class__}'
        assert isinstance(out, np.ndarray), error_msg

        return cv2.resize(out, self.output_size)

    def get_frame_kp(self, image):
        pass

    @staticmethod
    def normalize_alignment_kp(kp):
        pass

    def get_start_frame(self):
        pass

    def get_start_frame_kp(self):
        pass
