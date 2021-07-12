'''Module containing predictor using fsgan model'''
from typing import Tuple

from face_alignment import FaceAlignment, LandmarksType
from torch.nn import Module
import torch

from afy.frame_features import FrameFeatures
from afy.custom_typings import CV2Image
from afy.predictor import Predictor

# from fsgan.models.hopenet import Hopenet
from fsgan.data import landmark_transforms
from fsgan.utils.heatmap import LandmarkHeatmap
from fsgan.utils.obj_factory import obj_factory

BLEND_MODEL_PATH = '../weights/ijbc_msrunet_256_2_0_blending_v1.pth'
POSE_MODEL_PATH = '../weights/hopenet_robust_alpha1.pth'
REENACTMENT_MODEL_PATH = '../weights/ijbc_msrunet_256_2_0_reenactment_v1.pth'

PIL_TRANSFORMS = ('landmark_transforms.FaceAlignCrop', 'landmark_transforms.Resize(256)',
                  'landmark_transforms.Pyramids(2)')
TENSOR_TRANSFORMS = ('landmark_transforms.ToTensor()',
                     'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])')

def load_state_and_eval(model: Module, checkpoint: dict):
    '''
    Loads the state_dict contained in the checkpoint and set model to eval.
    '''
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def img_transforms(pil_transforms: Tuple[str], tensor_transforms: Tuple[str]):
    '''Create img_transforms based on pil and tensor transforms'''
    pil_transforms_arr = obj_factory(pil_transforms)
    tensor_transforms_arr = obj_factory(tensor_transforms)
    return landmark_transforms.ComposePyramids(
        pil_transforms_arr + tensor_transforms_arr
    )

class FSGANPredictor(Predictor):
    '''Predictor that uses a fsgan model as its backbone'''
    source = None

    def __init__(self, *_):
        super().__init__()
        self.aligner_2d = FaceAlignment(LandmarksType._2D, flip_input=False)
        self.aligner_3d = FaceAlignment(LandmarksType._3D, flip_input=False)
        self.landmarks2heatmaps = LandmarkHeatmap().to(self.device)
        self.gen_r = self._load_model(REENACTMENT_MODEL_PATH)
        self.gen_b = self._load_model(BLEND_MODEL_PATH)
        # self.gen_p = self._load_hopenet(POSE_MODEL_PATH)
        self.img_transforms = img_transforms(PIL_TRANSFORMS, TENSOR_TRANSFORMS)

    def _load_model(self, checkpoint_path: str):
        checkpoint: dict = torch.load(checkpoint_path)
        model: Module = obj_factory(checkpoint['arch']).to(self.device)
        return load_state_and_eval(model, checkpoint)

    # def _load_hopenet(self, checkpoint_path: str):
    #     model = Hopenet().to(self.device)
    #     checkpoint: dict = torch.load(checkpoint_path)
    #     return load_state_and_eval(model, checkpoint)

    def set_source_image(self, source_image: CV2Image):
        self.source = FrameFeatures(
            source_image, self.aligner_2d, self.aligner_3d, self.img_transforms
        )
