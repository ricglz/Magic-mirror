from typing import Tuple

from face_alignment import FaceAlignment, LandmarksType
import torch
from torch.nn import Module

from afy.predictor import Predictor
from fsgan.data import landmark_transforms
from fsgan.models.hopenet import Hopenet
from fsgan.utils.heatmap import LandmarkHeatmap
from fsgan.utils.obj_factory import obj_factory

BLEND_MODEL_PATH = '../weights/ijbc_msrunet_256_2_0_blending_v1.pth'
POSE_MODEL_PATH = '../weights/hopenet_robust_alpha1.pth'
REENACTMENT_MODEL_PATH = '../weights/ijbc_msrunet_256_2_0_reenactment_v1.pth'

pil_transforms1 = ('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                   'landmark_transforms.Pyramids(2)')
pil_transforms2 = ('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                   'landmark_transforms.Pyramids(2)')
tensor_transforms1 = ('landmark_transforms.ToTensor()',
                      'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])')
tensor_transforms2 = ('landmark_transforms.ToTensor()',
                      'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])')

def load_state_and_eval(model: Module, checkpoint: dict):
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def img_transforms(pil_transforms: Tuple[str], tensor_transforms: Tuple[str]):
    pil_transforms_arr = obj_factory(pil_transforms)
    tensor_transforms_arr = obj_factory(tensor_transforms)
    return landmark_transforms.ComposePyramids(
        pil_transforms_arr + tensor_transforms_arr
    )

class FSGANPredictor(Predictor):
    def __init__(self, *_):
        super().__init__()
        self.aligner = FaceAlignment(LandmarksType._3D, flip_input=False)
        self.landmarks2heatmaps = LandmarkHeatmap().to(self.device)
        self.gr = self._load_model(REENACTMENT_MODEL_PATH)
        self.gb = self._load_model(BLEND_MODEL_PATH)
        self.gp = self._load_hopenet(POSE_MODEL_PATH)
        self.img_transforms1 = img_transforms(pil_transforms1, tensor_transforms1)
        self.img_transforms2 = img_transforms(pil_transforms2, tensor_transforms2)

    def _load_model(self, checkpoint_path: str):
        checkpoint: dict = torch.load(checkpoint_path)
        model: Module = obj_factory(checkpoint['arch']).to(self.device)
        return load_state_and_eval(model, checkpoint)

    def _load_hopenet(self, checkpoint_path: str):
        model = Hopenet().to(self.device)
        checkpoint: dict = torch.load(checkpoint_path)
        return load_state_and_eval(model, checkpoint)
