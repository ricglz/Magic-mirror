from scipy.spatial import ConvexHull
import numpy as np

import torch
from torch.tensor import Tensor

from articulated.demo import load_checkpoints
from articulated.animate import get_animation_region_params
from articulated.modules.generator_optim import OcclusionAwareGenerator
from articulated.modules.keypoint_detector import KPDetector

def to_tensor(a: np.ndarray) -> Tensor:
    return Tensor(a[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2) / 255

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

    def reset_frames(self):
        pass

    def set_source_image(self, source_image):
        self.driving = to_tensor(np.array([source_image])).to(self.device)
        self.driving_region_params = self.region_predictor(self.driving)

    def predict(self, driving_frame):
        assert self.driving_region_params is not None, "call set_source_image()"

        with torch.no_grad():
            source = to_tensor(driving_frame).to(self.device)
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

            out = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
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
