'''Module containing the local predictor class'''
from facenet_pytorch import MTCNN
from PIL import Image
from scipy.spatial import ConvexHull
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image

from afy.face_swap import swap_faces
from afy.magic_mirror import MagicMirror
from afy.utils import Logger
from articulated.animate import get_animation_region_params
from articulated.demo import load_checkpoints

mtcnn = MTCNN()

def to_tensor(a: np.ndarray):
    return torch.tensor(a[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

def to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img) / 255

def pil_to_cv2(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def extract_face(image: Image.Image, box) -> Image.Image:
    margin = 25
    box[0] -= margin
    box[1] -= margin
    box[2] += margin
    box[3] += margin
    return image.crop(box).resize((256, 256))

def get_box_and_landmarks(image):
    box, _, landmarks = mtcnn.detect(image, True)
    box = box[0]
    landmarks = np.array(landmarks[0], np.int32)
    return box, landmarks

def get_face(image_numpy: np.ndarray):
    with torch.no_grad():
        image = Image.fromarray(cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB))
        box, landmarks = get_box_and_landmarks(image)
        face = to_tensor(to_numpy(extract_face(image, box)))
        to_pil_image(face[0]).save('face.jpg')
        return face, landmarks

class PredictorLocal:
    output_size = (512, 512)

    def __init__(self, config_path: str, checkpoint_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        networks = load_checkpoints(config_path, checkpoint_path, self.device == 'cpu')
        self.generator, self.region_predictor, self.avd_network = networks
        self.driving = None
        self.driving_region_params = None
        self.magic_mirror = MagicMirror()

    def reset_frames(self):
        pass

    def set_source_image(self, source_image):
        self.magic_mirror.reset_tic()
        self.driving = get_face(source_image)[0].to(self.device)
        self.driving_region_params = self.region_predictor(self.driving)

    def _predict(self, driving_frame: np.ndarray):
        with torch.no_grad():
            source = get_face(driving_frame)
            source, landmarks = get_face(driving_frame)
            source_img_data = driving_frame, landmarks

            source_region_params = self.region_predictor(source)

            new_region_params = get_animation_region_params(
                self.driving_region_params,
                source_region_params,
                source_region_params,
                avd_network=self.avd_network,
                mode='avd'
            )

            modified_face = self.generator(
                self.driving,
                source_region_params=self.driving_region_params,
                driving_region_params=new_region_params
            )['prediction'][0]
            out = modified_face

            modified_face_img = to_pil_image(modified_face)
            _, modified_landmarks = get_box_and_landmarks(modified_face_img)
            modified_img_data = pil_to_cv2(modified_face_img), modified_landmarks
            out = swap_faces(source_img_data, modified_img_data)

            return out

    def predict(self, driving_frame: np.ndarray):
        assert self.driving_region_params is not None, "call set_source_image()"

        if self.magic_mirror.should_predict():
            out = self._predict(driving_frame)
        else:
            out = driving_frame

        error_msg = f'Expected out to be np.ndarray, got {out.__class__}'
        assert isinstance(out, np.ndarray), error_msg

        out = Image.fromarray(out).resize(self.output_size)
        out.save('out_image_pil.jpg')

        out = pil_to_cv2(out)
        cv2.imwrite('out_image_cv2.jpg', out)

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
