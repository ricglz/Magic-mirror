"""
Originally from: https://github.com/hay/facetool/blob/master/facetool/faceswap.py
"""
from enum import Enum
from typing import Optional

from face_alignment import FaceAlignment
import cv2
import numpy

from afy.custom_typings import CV2Image, BBoxes
from afy.eds_swap import get_swap_function as get_eds_function
from afy.triangulation_swap import swap_imgs as triangulation_swap

class SwapMethod(Enum):
    '''Enum to organize the swap methods for faces'''
    EDS = 'EDS'
    TRIANGULATION = 'TRIANGULATION'

class Faceswap:
    '''Face swap function'''
    def __init__(
        self,
        aligner: FaceAlignment,
        swap_method = SwapMethod.TRIANGULATION,
        feather=35,
        blur=2.2
    ):
        self.aligner = aligner
        self.landmark_hashes = {}
        if swap_method == SwapMethod.EDS:
            self.swap_function = get_eds_function(blur, feather)
        elif swap_method == SwapMethod.TRIANGULATION:
            self.swap_function = triangulation_swap

    def get_bboxes(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.aligner.face_detector.detect_from_image(img)

    def get_landmarks(self, img, bboxes=None):
        # This is by far the slowest part of the whole algorithm, so we
        # cache the landmarks if the image is the same, especially when
        # dealing with videos this makes things twice as fast
        img_hash = str(abs(hash(img.data.tobytes())))

        if img_hash in self.landmark_hashes:
            return self.landmark_hashes[img_hash]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        points = self.aligner.get_landmarks_from_image(img, bboxes)
        landmarks = points[0].astype(int)

        # Save to image cache
        self.landmark_hashes[img_hash] = landmarks

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

        out = self.swap_function(head, face, landmarks1, landmarks2)
        return out.astype(int)
