'''Module containing the local predictor class'''
from face_alignment import FaceAlignment, LandmarksType
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import cv2
import numpy as np
import torch

from afy.custom_typings import BBox, CV2Image
from afy.face_swap import Faceswap
from afy.magic_mirror import MagicMirror
from articulated.animate import get_animation_region_params
from articulated.demo import load_checkpoints

aligner = FaceAlignment(
    LandmarksType._2D,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    face_detector='blazeface',
)

def to_tensor(a: np.ndarray):
    '''Creates tensor of numpy array of an image'''
    return torch.tensor(a[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)

def to_numpy(img: Image.Image) -> np.ndarray:
    '''Converts pil image to numpy representation.'''
    return np.array(img) / 255

def pil_to_cv2(img: Image.Image) -> CV2Image:
    '''Converts pil image to a cv2 image'''
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def extract_face(image: Image.Image, box: BBox) -> Image.Image:
    margin = 25
    box[0] -= margin
    box[1] -= margin
    box[2] += margin
    box[3] += margin
    return image.crop(box).resize((256, 256))

@torch.no_grad()
def get_face(image_numpy: CV2Image):
    color_img = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    bbox = aligner.face_detector.detect_from_image(color_img.copy())[0]
    pil_image = Image.fromarray(color_img)
    face = to_tensor(to_numpy(extract_face(pil_image, bbox)))
    return face, bbox

class PredictorLocal:
    output_size = (512, 512)

    def __init__(self, config_path: str, checkpoint_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        networks = load_checkpoints(config_path, checkpoint_path, self.device == 'cpu')
        self.generator, self.region_predictor, self.avd_network = networks
        self.driving = None
        self.driving_region_params = None
        self.magic_mirror = MagicMirror()
        self.face_swapper = Faceswap(aligner)

    def reset_frames(self):
        pass

    @torch.no_grad
    def set_source_image(self, source_image: CV2Image):
        self.magic_mirror.reset_tic()
        self.driving = get_face(source_image)[0].to(self.device)
        self.driving_region_params = self.region_predictor(self.driving)

    @torch.no_grad
    def _face_swap(
        self,
        source: CV2Image,
        bbox: BBox,
        modified_face: torch.Tensor
    ):
        cv2_modified_face = pil_to_cv2(to_pil_image(modified_face))
        return self.face_swapper.faceswap(source, cv2_modified_face, [bbox])

    @torch.no_grad
    def _predict(self, driving_frame: CV2Image):
        source, bbox = get_face(driving_frame)

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
        # out = modified_face

        out = self._face_swap(source, bbox, modified_face)

        return out

    def predict(self, driving_frame: CV2Image):
        assert self.driving_region_params is not None, "call set_source_image()"

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
