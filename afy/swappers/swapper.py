'''Module containing the Swapper abstract class'''
from abc import ABC, abstractmethod
import numpy as np
from afy.custom_typings import CV2Image

class Swapper(ABC):
    '''Swapper abstract class'''
    @classmethod
    def __subclasshook__(cls, subclass):
        if hasattr(subclass, 'swap_imgs') and callable(subclass.swap_imgs):
            return True
        return NotImplemented

    @abstractmethod
    def swap_imgs(
        self,
        im1: CV2Image,
        im2: CV2Image,
        landmarks1: np.ndarray,
        landmarks2: np.ndarray,
    ) -> CV2Image:
        '''Swap the images using their corresponding landmarks'''
        raise NotImplementedError
