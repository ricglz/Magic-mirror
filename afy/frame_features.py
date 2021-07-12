'''Convinient object for managing frame features'''
from typing import Tuple

from face_alignment import FaceAlignment
import numpy as np

from afy.custom_typings import CV2Image
from fsgan.utils.bbox_utils import get_main_bbox

Features3D = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def get_3D_features(
    rgb_frame: CV2Image,
    aligner_2d: FaceAlignment,
    aligner_3d: FaceAlignment,
) -> Features3D:
    detected_faces = aligner_2d.face_detector.detect_from_image(rgb_frame.copy())

    # Skip current frame there if no faces were detected
    if len(detected_faces) == 0:
        return None
    curr_bbox = get_main_bbox(np.array(detected_faces)[:, :4], rgb_frame.shape[:2])
    detected_faces = [curr_bbox]

    curr_landmarks = aligner_2d.get_landmarks(rgb_frame, detected_faces)[0]
    curr_landmarks_3d = aligner_3d.get_landmarks(rgb_frame, detected_faces)[0]

    # Convert bounding boxes format from [min, max] to [min, size]
    curr_bbox[2:] = curr_bbox[2:] - curr_bbox[:2] + 1
    return curr_bbox, curr_landmarks, curr_landmarks_3d

class FrameFeatures():
    def __init__(self, frame, aligner_2d, aligner_3d, img_transforms):
        frame_rgb = frame[..., ::-1]
        features = get_3D_features(frame_rgb, aligner_2d, aligner_3d)
        transformed = img_transforms(frame_rgb, features[1], features[0])
        self.tensor, self.landmarks, _ = transformed
        _, self.landmarks_3d, _ = img_transforms(
            frame_rgb, features[2], features[0]
        )
