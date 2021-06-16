from facenet_pytorch import MTCNN
from PIL import Image
from scipy.spatial import ConvexHull
import numpy as np

import torch
from torch.tensor import Tensor
from torchvision.transforms.functional import to_pil_image

from articulated.demo import load_checkpoints
from articulated.animate import get_animation_region_params

from afy.magic_mirror import MagicMirror

def to_tensor(a: np.ndarray) -> Tensor:
    return Tensor(a[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2) / 255

def to_numpy(img: Image.Image):
    return np.array(img.convert('RGB')) / 255.0

def extract_face(image: Image.Image, box) -> Image.Image:
    margin = 25
    box[0] -= margin
    box[1] -= margin
    box[2] += margin
    box[3] += margin
    return image.crop(box).resize((256, 256))

def get_face(image_numpy: np.ndarray, mtcnn: MTCNN):
    with torch.no_grad():
        image = to_pil_image(to_tensor(image_numpy)).resize((512, 512))
        box = mtcnn.detect(image)[0][0]
        face = to_tensor(to_numpy(extract_face(image, box)))
        return face, box

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
        self.driving = get_face(source_image[0], self.mtcnn)[0].to(self.device)
        self.driving_region_params = self.region_predictor(self.driving)

    def _predict(self, source):
        with torch.no_grad():
            source_region_params = self.region_predictor(source)

            new_region_params = get_animation_region_params(
                self.driving_region_params,
                source_region_params,
                source_region_params,
                avd_network=self.avd_network,
                mode='avd'
            )
            out = self.generator(
                self.driving,
                source_region_params=self.driving_region_params,
                driving_region_params=new_region_params
            )

            return out


    def predict(self, driving_frame):
        assert self.driving_region_params is not None, "call set_source_image()"

        source, _ = get_face(driving_frame[0], self.mtcnn)
        source = source.to(self.device)

        if self.magic_mirror.should_predict():
            out = self._predict(source)['prediction']
        else:
            out = source

        out = np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])[0]
        out = (np.clip(out, 0, 1) * 255).astype(np.uint8)

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
