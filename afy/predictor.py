from abc import ABC, abstractmethod

import torch

from afy.custom_typings import CV2Image
from afy.magic_mirror import MagicMirror

class Predictor(ABC):
    '''Swapper abstract class'''
    def __init__(self, *_):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.magic_mirror = MagicMirror()

    @classmethod
    def __subclasshook__(cls, subclass):
        if hasattr(subclass, 'set_source_image') and callable(subclass.set_source_image) and \
           hasattr(subclass, 'predict') and callable(subclass.predict):
            return True
        return NotImplemented

    def reset_frames(self):
        pass

    @abstractmethod
    def set_source_image(self, source_image: CV2Image):
        raise NotImplementedError

    @abstractmethod
    def predict(self, driving_frame: CV2Image):
        raise NotImplementedError

    def get_frame_kp(self, image):
        pass

    @staticmethod
    def normalize_alignment_kp(kp):
        pass

    def get_start_frame(self):
        pass

    def get_start_frame_kp(self):
        pass
