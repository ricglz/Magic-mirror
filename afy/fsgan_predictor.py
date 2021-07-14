'''Module containing predictor using fsgan model'''
from typing import Dict, Optional, Tuple

from face_alignment import FaceAlignment, LandmarksType
from torch.nn import Module
import numpy as np
import torch
import torch.nn.functional as F

from afy.custom_typings import CV2Image
from afy.frame_features import FrameFeatures
from afy.predictor import Predictor
from afy.utils import hash_numpy_array

# from fsgan.models.hopenet import Hopenet
from fsgan.data import landmark_transforms
from fsgan.utils.estimate_pose import rigid_transform_3D
from fsgan.utils.heatmap import LandmarkHeatmap
from fsgan.utils.obj_factory import obj_factory

BLEND_MODEL_PATH = 'weights/ijbc_msrunet_256_2_0_blending_v1.pth'
POSE_MODEL_PATH = 'weights/hopenet_robust_alpha1.pth'
REENACTMENT_MODEL_PATH = 'weights/ijbc_msrunet_256_2_0_reenactment_v1.pth'

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

def transfer_mask(img1, img2, mask):
    mask = mask.view(mask.shape[0], 1, mask.shape[1], mask.shape[2]).repeat(1, 3, 1, 1).float()
    return img1 * mask + img2 * (1 - mask)

def create_pyramid(img, n=1):
    if isinstance(img, (list, tuple)):
        return img

    pyd = [img]
    for _ in range(n - 1):
        elem = F.avg_pool2d(pyd[-1], 3, stride=2, padding=1, count_include_pad=False)
        pyd.append(elem)

    return pyd

def unnormalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def tensor2bgr(img_tensor):
    output_img = unnormalize(img_tensor.clone(), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    output_img = output_img.squeeze().permute(1, 2, 0).cpu().numpy()
    output_img = np.round(output_img[:, :, ::-1] * 255).astype('uint8')

    return output_img

def get_transformed_landmarks(source: FrameFeatures, out_pts: np.ndarray):
    '''Transfer mouth points only.'''
    source_landmarks_np = source.landmarks[0].cpu().numpy().copy()
    mouth_pts = out_pts[48:, :2] - out_pts[48:, :2].mean(axis=0) + source_landmarks_np[48:, :].mean(axis=0)
    transformed_landmarks = source_landmarks_np
    transformed_landmarks[48:, :] = mouth_pts
    return transformed_landmarks

class FSGANPredictor(Predictor):
    '''Predictor that uses a fsgan model as its backbone'''
    cached_frame_features: Dict[str, FrameFeatures] = {}
    target: Optional[FrameFeatures] = None

    def __init__(self, swap_face: bool, verbose: bool, **_):
        super().__init__(swap_face, 'fsgan_predictor', verbose)
        self.aligner_2d = FaceAlignment(LandmarksType._2D, face_detector='blazeface')
        self.aligner_3d = FaceAlignment(LandmarksType._3D, face_detector='blazeface')
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

    def _get_frame_features(self, frame: CV2Image):
        frame_hash = hash_numpy_array(frame)
        if frame_hash in self.cached_frame_features:
            return self.cached_frame_features[frame_hash]
        frame_features = FrameFeatures(
            frame, self.aligner_2d, self.aligner_3d, self.img_transforms
        )
        self.cached_frame_features[frame_hash] = frame_features
        return frame_features

    def _set_source_image(self, source_image: CV2Image):
        self.target = self._get_frame_features(source_image)

    def _get_out_pts(self, source: FrameFeatures) -> np.ndarray:
        R, t = rigid_transform_3D(
            self.target.landmarks_3d[0].numpy(), source.landmarks_3d[0].numpy()
        )
        pts = self.target.landmarks_3d[0].numpy().transpose()
        out_pts = R @ pts
        translation = np.tile(t, (out_pts.shape[1], 1)).transpose()
        out_pts += translation
        return out_pts.transpose()

    def _create_heatmap_pyramids(self, transformed_landmarks):
        transformed_landmarks_tensor = torch.from_numpy(transformed_landmarks).unsqueeze(0).to(self.device)
        transformed_hm_tensor = self.landmarks2heatmaps(transformed_landmarks_tensor)
        interpolation = F.interpolate(
            transformed_hm_tensor,
            scale_factor=0.5,
            mode='bilinear',
            align_corners=False
        )
        return [transformed_hm_tensor, interpolation]

    def _face_reenactment(self, source: FrameFeatures, transformed_hm_tensor_pyd):
        reenactment_input_tensor = []
        for j, _ in enumerate(source.tensor):
            source.tensor[j] = source.tensor[j].unsqueeze(0).to(self.device)
            reenactment_input_tensor.append(
                torch.cat((source.tensor[j], transformed_hm_tensor_pyd[j]), dim=1))
        return self.gen_r(reenactment_input_tensor)

    def _predict(self, driving_frame: CV2Image):
        source = self._get_frame_features(driving_frame)
        out_pts = self._get_out_pts(source)
        transformed_landmarks = get_transformed_landmarks(source, out_pts)
        transformed_hm_tensor_pyd = self._create_heatmap_pyramids(transformed_landmarks)
        reenactment_img_tensor, reenactment_seg_tensor = self._face_reenactment(
            source, transformed_hm_tensor_pyd
        )

        # Transfer reenactment to original image
        source_orig_tensor = source.tensor[0].to(self.device)
        face_mask_tensor = reenactment_seg_tensor.argmax(1) == 1
        transfer_tensor = transfer_mask(
            reenactment_img_tensor, source_orig_tensor, face_mask_tensor
        )

        # Blend the transfer image with the source image
        blend_input_tensor = torch.cat(
            (transfer_tensor, source_orig_tensor, face_mask_tensor.unsqueeze(1).float()), dim=1)
        blend_input_tensor_pyd = create_pyramid(blend_input_tensor, len(source.tensor))
        blend_tensor = self.gen_b(blend_input_tensor_pyd)

        return tensor2bgr(blend_tensor)
