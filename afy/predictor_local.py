from facenet_pytorch import MTCNN
from PIL import Image
from scipy.spatial import ConvexHull
from PIL import Image
import numpy as np

import torch
from torch.tensor import Tensor
from torchvision.transforms.functional import to_pil_image

from afy.face_swap import swap_faces
from afy.magic_mirror import MagicMirror
from afy.utils import Logger
from articulated.animate import get_animation_region_params
from articulated.demo import load_checkpoints

log = Logger('./var/log/predictor_local.log')

def to_tensor(a: np.ndarray) -> Tensor:
    return Tensor(a[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

def to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img.convert('RGB')) / 255

def from_numpy_to_pil(array: np.ndarray):
    return Image.fromarray(array)

def extract_face(image: Image.Image, box) -> Image.Image:
    log('extracting face', important=True)
    margin = 25
    box[0] -= margin
    box[1] -= margin
    box[2] += margin
    box[3] += margin
    return image.crop(box).resize((256, 256))

def get_box_and_landmarks(image, mtcnn: MTCNN):
    log('getting box and landmarks', important=True)
    box, _, landmarks = mtcnn.detect(image, True)
    box = box[0]
    landmarks = np.array(landmarks[0], np.int32)
    return box, landmarks

def get_face(image_numpy: np.ndarray, mtcnn: MTCNN):
    log('Getting face', important=True)
    with torch.no_grad():
        image = to_pil_image(to_tensor(image_numpy)[0]).resize((512, 512))
        box, landmarks = get_box_and_landmarks(image, mtcnn)
        face = to_tensor(to_numpy(extract_face(image, box)))
        return face, landmarks

class PredictorLocal:
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        relative=False,
        adapt_movement_scale=False,
        device=None,
        enc_downscale=1
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        networks = load_checkpoints(
                config_path, checkpoint_path, device == 'cpu'
        )
        self.generator, self.region_predictor, self.avd_network = networks
        self.driving = None
        self.driving_region_params = None
        self.mtcnn = MTCNN()
        self.magic_mirror = MagicMirror()

    def reset_frames(self):
        pass

    def set_source_image(self, source_image):
        self.magic_mirror.reset_tic()
        self.driving = get_face(source_image, self.mtcnn)[0].to(self.device)
        self.driving_region_params = self.region_predictor(self.driving)

    def _predict(self, driving_frame):
        with torch.no_grad():
            source, landmarks = get_face(driving_frame, self.mtcnn)
            source = source.to(self.device)
            source_img_data = driving_frame, landmarks

            log('Calculating source region params', important=True)
            source_region_params = self.region_predictor(source)

            log('Calculating new region params', important=True)
            new_region_params = get_animation_region_params(
                self.driving_region_params,
                source_region_params,
                source_region_params,
                avd_network=self.avd_network,
                mode='avd'
            )

            log('Calculating modified face', important=True)
            modified_face = self.generator(
                self.driving,
                source_region_params=self.driving_region_params,
                driving_region_params=new_region_params
            )['prediction'][0]

            log('Doing face-swapping', important=True)
            modified_face_img = to_pil_image(modified_face).resize((512, 512))
            _, modified_landmarks = get_box_and_landmarks(modified_face_img, self.mtcnn)
            modified_img_data = np.array(modified_face_img), modified_landmarks
            out = swap_faces(source_img_data, modified_img_data)

            return out

    def predict(self, driving_frame):
        assert self.driving_region_params is not None, "call set_source_image()"

        if self.magic_mirror.should_predict():
            out = self._predict(driving_frame)
        else:
            out = driving_frame

        out = to_numpy(from_numpy_to_pil(out).resize((512, 512)))

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
