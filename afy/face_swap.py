"""
Originally from: https://github.com/hay/facetool/blob/master/facetool/faceswap.py
"""
from enum import IntEnum
from typing import Optional

from face_alignment import FaceAlignment
import cv2

from afy.custom_typings import CV2Image, BBoxes
from afy.swappers import *
from afy.utils import np_to_hash

class SwapMethod(IntEnum):
    '''Enum to organize the swap methods for faces'''
    EDS = 0
    TRIANGULATION = 1
    POISSON = 2

    @staticmethod
    def parse_str(value: str):
        '''Parses a string value to a SwapMethod enum'''
        if value == 'eds':
            return SwapMethod.EDS
        if value == 'triangulation':
            return SwapMethod.TRIANGULATION
        return SwapMethod.POISSON

class Faceswap:
    '''Face swap function'''
    available_swappers = [EDSSwapper, TriangulationSwapper, PoissonSwapper]

    def __init__(
        self,
        aligner: FaceAlignment,
        swap_method = SwapMethod.POISSON,
        **kwargs,
    ):
        self.aligner = aligner
        self.cached_landmarks = {}
        cls = self.available_swappers[int(swap_method)]
        self.swapper: Swapper = cls(**kwargs)

    def get_bboxes(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.aligner.face_detector.detect_from_image(img)

    def get_landmarks(self, img, bboxes=None):
        # This is by far the slowest part of the whole algorithm, so we
        # cache the landmarks if the image is the same, especially when
        # dealing with videos this makes things twice as fast
        img_hash = np_to_hash(img)

        if img_hash in self.cached_landmarks:
            return self.cached_landmarks[img_hash]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        points = self.aligner.get_landmarks_from_image(img, bboxes)
        landmarks = points[0].astype(int)

        # Save to image cache
        self.cached_landmarks[img_hash] = landmarks

        return landmarks

    def faceswap(
        self,
        head: CV2Image,
        face: CV2Image,
        head_bboxes: Optional[BBoxes]=None,
        face_bboxes: Optional[BBoxes]=None
    ):
        '''Inserts the face into the head.'''
        try:
            landmarks1 = self.get_landmarks(head.copy(), head_bboxes)
            landmarks2 = self.get_landmarks(face.copy(), face_bboxes)
        except ValueError:
            return head

        out = self.swapper.swap_imgs(head, face, landmarks1, landmarks2)
        return out.astype(int)
