from abc import ABC, abstractmethod

import numpy as np
import torch

from afy.custom_typings import CV2Image
from afy.image_logger import ImageLogger
from afy.magic_mirror import MagicMirror
from afy.utils import Logger, resize

class Predictor(ABC):
    '''Swapper abstract class'''

    def __init__(
        self,
        cls_name: str,
        swap_face: bool,
        verbose: bool,
        resolution: int,
        **_,
    ):
        self.swap_face = swap_face
        self.verbose = verbose
        self.logger = Logger(f'./var/log/{cls_name}.log', verbose)
        self.image_logger = ImageLogger(f'./imgs/{cls_name}', verbose)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.magic_mirror = MagicMirror()
        self.output_size = (resolution, resolution)

    @classmethod
    def __subclasshook__(cls, subclass):
        if hasattr(subclass, '_set_source_image') and callable(subclass._set_source_image) and \
           hasattr(subclass, '_predict') and callable(subclass._predict):
            return True
        return NotImplemented

    def reset_frames(self):
        pass

    @abstractmethod
    def _set_source_image(self, source_image: CV2Image):
        raise NotImplementedError

    def set_source_image(self, source_image: CV2Image):
        self.magic_mirror.reset_tic()
        self._set_source_image(source_image)

    @abstractmethod
    def _predict(self, driving_frame: CV2Image):
        raise NotImplementedError

    def predict(self, driving_frame: CV2Image):
        if self.magic_mirror.should_predict():
            out = self._predict(driving_frame)
            self.logger(out[:, 0, 0])
        else:
            out = driving_frame

        error_msg = f'Expected out to be np.ndarray, got {out.__class__}'
        assert isinstance(out, np.ndarray), error_msg

        out = resize(out, self.output_size)

        if out.dtype is not np.dtype(np.uint8) or out.dtype is not np.dtype('int'):
            out = (out * 255).astype(int)

        return out

    def get_frame_kp(self, image):
        pass

    @staticmethod
    def normalize_alignment_kp(kp):
        pass

    def get_start_frame(self):
        pass

    def get_start_frame_kp(self):
        pass
