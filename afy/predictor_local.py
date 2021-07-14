'''Module containing the local predictor class'''
from face_alignment import FaceAlignment, LandmarksType
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import cv2
import numpy as np
import torch

from afy.custom_typings import BBox, CV2Image
from afy.face_swap import Faceswap
from afy.predictor import Predictor
from afy.utils import Logger

from articulated.animate import get_animation_region_params
from articulated.demo import load_checkpoints

MODEL_SIZE = (256, 256)
log = Logger('./var/log/predictor_local.log')

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
    if isinstance(box, np.ndarray):
        box = box[:4]
    box[0] -= margin
    box[1] -= margin
    box[2] += margin
    box[3] += margin
    return image.crop(box).resize(MODEL_SIZE)

@torch.no_grad()
def get_face(image_numpy: CV2Image, aligner: FaceAlignment):
    color_img = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    bbox = aligner.face_detector.detect_from_image(color_img.copy())[0]
    pil_image = Image.fromarray(color_img)
    face = to_tensor(to_numpy(extract_face(pil_image, bbox)))
    return face, bbox

class PredictorLocal(Predictor):
    driving = None
    driving_region_params = None

    def __init__(
        self,
        swap_face: bool,
        config_path: str,
        checkpoint_path: str,
    ):
        super().__init__()
        self.networks = load_checkpoints(
            config_path, checkpoint_path, self.device == 'cpu'
        )
        self.aligner = FaceAlignment(
            LandmarksType._2D, device=self.device, face_detector='blazeface',
        )

        self.swap_face = swap_face
        self.face_swapper = Faceswap(self.aligner)

    @property
    def generator(self):
        return self.networks[0]

    @property
    def region_predictor(self):
        return self.networks[1]

    @property
    def avd_network(self):
        return self.networks[2]

    def _prepare_img(self, img: CV2Image):
        bbox = None
        if self.swap_face:
            parsed_img, bbox = get_face(img, self.aligner)
        else:
            parsed_img = to_tensor(cv2.resize(img[..., ::-1], MODEL_SIZE))
        log(parsed_img, bbox is None, important=True)
        return parsed_img.to(self.device), bbox

    @torch.no_grad()
    def _set_source_image(self, source_image: CV2Image):
        self.driving = self._prepare_img(source_image)[0]
        self.driving_region_params = self.region_predictor(self.driving)

    @torch.no_grad()
    def _face_swap(
        self,
        source: CV2Image,
        bbox: BBox,
        modified_face: torch.Tensor
    ):
        # cv2_modified_face = pil_to_cv2(to_pil_image(modified_face))[...,::-1]
        return self.face_swapper.faceswap(source, modified_face, [bbox])

    @torch.no_grad()
    def _predict(self, driving_frame: CV2Image):
        source, bbox = self._prepare_img(driving_frame)

        log('Source region params', important=True)
        source_region_params = self.region_predictor(source)

        log('New region params', important=True)
        new_region_params = get_animation_region_params(
            self.driving_region_params,
            source_region_params,
            source_region_params,
            avd_network=self.avd_network,
            mode='avd'
        )

        log('Generator', important=True)
        out = self.generator(
            self.driving,
            source_region_params=self.driving_region_params,
            driving_region_params=new_region_params
        )['prediction'][0]
        out = pil_to_cv2(to_pil_image(out))[...,::-1]

        if self.swap_face:
            log('Faceswap', important=True)
            out = self._face_swap(driving_frame, bbox, out)

        return out
