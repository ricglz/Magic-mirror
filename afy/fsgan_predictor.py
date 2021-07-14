'''Module containing predictor using fsgan model'''
from typing import Dict, Optional, Tuple

from face_alignment import FaceAlignment, LandmarksType
from torch.nn import Module
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from afy.custom_typings import CV2Image
from afy.frame_features import FrameFeatures
from afy.predictor import Predictor
from afy.swappers.constants import EYES_BROWS_POINTS, MOUTH_POINTS
from afy.utils import hash_numpy_array

# from fsgan.models.hopenet import Hopenet
from fsgan.data import landmark_transforms
from fsgan.utils.estimate_pose import rigid_transform_3D
from fsgan.utils.heatmap import LandmarkHeatmap
from fsgan.utils.obj_factory import obj_factory

BLEND_MODEL_PATH = 'weights/ijbc_msrunet_256_2_0_blending_v1.pth'
POSE_MODEL_PATH = 'weights/hopenet_robust_alpha1.pth'
REENACTMENT_MODEL_PATH = 'weights/ijbc_msrunet_256_2_0_reenactment_v1.pth'

REENACTMENT_ARCH = \
    'res_unet_split.MultiScaleResUNet(in_nc=71,out_nc=(3,3),flat_layers=(2,0,2,3),ngf=128)'

PIL_TRANSFORMS = ('landmark_transforms.Resize(256)',
                  'landmark_transforms.Pyramids(2)')
TENSOR_TRANSFORMS = ('landmark_transforms.ToTensor()',
                     'transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])')

POINTS_TO_TRANSFORM = EYES_BROWS_POINTS + MOUTH_POINTS

def load_state_and_eval(model: Module, checkpoint: dict):
    '''
    Loads the state_dict contained in the checkpoint and set model to eval.
    '''
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

@torch.no_grad()
def img_transforms(pil_transforms: Tuple[str], tensor_transforms: Tuple[str]):
    '''Create img_transforms based on pil and tensor transforms'''
    pil_transforms_arr = obj_factory(pil_transforms)
    tensor_transforms_arr = obj_factory(tensor_transforms)
    return landmark_transforms.ComposePyramids(
        [to_pil_image] + pil_transforms_arr + tensor_transforms_arr
    )

def transfer_mask(img1, img2, mask):
    mask = mask.view(mask.shape[0], 1, mask.shape[1], mask.shape[2]).repeat(1, 3, 1, 1).float()
    return img1 * mask + img2 * (1 - mask)

@torch.no_grad()
def create_pyramid(img, n=1):
    if isinstance(img, (list, tuple)):
        return img

    pyd = [img]
    for _ in range(n - 1):
        elem = F.avg_pool2d(pyd[-1], 3, stride=2, padding=1, count_include_pad=False)
        pyd.append(elem)

    return pyd

@torch.no_grad()
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

@torch.no_grad()
def tensor2bgr(img_tensor):
    output_img = unnormalize(img_tensor.clone(), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    if len(output_img.size()) > 3:
        output_img = output_img[0]
    output_img = output_img.permute(1, 2, 0).cpu().numpy()
    output_img = np.round(output_img[:, :, ::-1] * 255).astype('uint8')

    return output_img

def get_transformed_landmarks(source: FrameFeatures, out_pts: np.ndarray):
    '''Transfer mouth points only.'''
    source_landmarks_np = source.landmarks[0].cpu().numpy().copy()
    mouth_pts = out_pts[MOUTH_POINTS, :2] - \
                out_pts[MOUTH_POINTS, :2].mean(axis=0) + \
                source_landmarks_np[MOUTH_POINTS, :].mean(axis=0)
    transformed_landmarks = source_landmarks_np
    transformed_landmarks[MOUTH_POINTS, :] = mouth_pts
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
        self.gen_r = self._load_model(REENACTMENT_MODEL_PATH, REENACTMENT_ARCH)
        self.gen_b = self._load_model(BLEND_MODEL_PATH)
        # self.gen_p = self._load_hopenet(POSE_MODEL_PATH)
        self.img_transforms = img_transforms(PIL_TRANSFORMS, TENSOR_TRANSFORMS)

    def _load_model(self, checkpoint_path: str, arch=None):
        checkpoint: dict = torch.load(checkpoint_path)
        self.logger(checkpoint_path, 'arch' in checkpoint)
        if arch is None:
            arch = checkpoint['arch']
        model: Module = obj_factory(arch).to(self.device)
        return load_state_and_eval(model, checkpoint)

    # def _load_hopenet(self, checkpoint_path: str):
    #     model = Hopenet().to(self.device)
    #     checkpoint: dict = torch.load(checkpoint_path)
    #     return load_state_and_eval(model, checkpoint)

    @torch.no_grad()
    def _get_frame_features(self, frame: CV2Image):
        self.logger('get frame features')
        frame_hash = hash_numpy_array(frame)
        if frame_hash in self.cached_frame_features:
            return self.cached_frame_features[frame_hash]
        frame_features = FrameFeatures(
            frame, self.aligner_2d, self.aligner_3d, self.img_transforms
        )
        self.cached_frame_features[frame_hash] = frame_features
        return frame_features

    def _set_source_image(self, source_image: CV2Image):
        self.logger('Set source image')
        self.target = self._get_frame_features(source_image)

    @torch.no_grad()
    def _get_out_pts(self, source: FrameFeatures) -> np.ndarray:
        self.logger('get out pts')
        R, t = rigid_transform_3D(
            self.target.landmarks_3d[0].numpy(), source.landmarks_3d[0].numpy()
        )
        pts = self.target.landmarks_3d[0].numpy().transpose()
        out_pts = R @ pts
        translation = np.tile(t, (out_pts.shape[1], 1)).transpose()
        out_pts += translation
        return out_pts.transpose()

    @torch.no_grad()
    def _create_heatmap_pyramids(self, transformed_landmarks):
        self.logger('create heatmap pyramids')
        transformed_landmarks_tensor = torch.from_numpy(transformed_landmarks).unsqueeze(0).to(self.device)
        transformed_hm_tensor = self.landmarks2heatmaps(transformed_landmarks_tensor)
        interpolation = F.interpolate(
            transformed_hm_tensor,
            scale_factor=0.5,
            mode='bilinear',
            align_corners=False
        )
        return [transformed_hm_tensor, interpolation]

    @torch.no_grad()
    def _face_reenactment(self, source: FrameFeatures, transformed_hm_tensor_pyd):
        self.logger('face reenactment')
        reenactment_input_tensor = []
        for j, _ in enumerate(source.tensor):
            source.tensor[j] = source.tensor[j].unsqueeze(0).to(self.device)
            reenactment_input_tensor.append(
                torch.cat((source.tensor[j], transformed_hm_tensor_pyd[j]), dim=1))
        return self.gen_r(reenactment_input_tensor)

    @torch.no_grad()
    def _predict(self, driving_frame: CV2Image):
        self.logger('Predict')
        source = self._get_frame_features(driving_frame)
        self.image_logger.save_cv2(tensor2bgr(source.tensor[0]))
        out_pts = self._get_out_pts(source)
        transformed_landmarks = get_transformed_landmarks(source, out_pts)
        transformed_hm_tensor_pyd = self._create_heatmap_pyramids(transformed_landmarks)
        reenactment_img_tensor, reenactment_seg_tensor = self._face_reenactment(
            source, transformed_hm_tensor_pyd
        )
        self.image_logger.save_cv2(tensor2bgr(reenactment_img_tensor))

        # Transfer reenactment to original image
        self.logger('transfer reenactment')
        source_orig_tensor = source.tensor[0].to(self.device)
        face_mask_tensor = reenactment_seg_tensor.argmax(1) == 1
        transfer_tensor = transfer_mask(
            reenactment_img_tensor, source_orig_tensor, face_mask_tensor
        )
        self.image_logger.save_cv2(tensor2bgr(transfer_tensor))

        self.logger('Blend transfer with source')
        # Blend the transfer image with the source image
        blend_input_tensor = torch.cat(
            (transfer_tensor, source_orig_tensor, face_mask_tensor.unsqueeze(1).float()), dim=1)
        blend_input_tensor_pyd = create_pyramid(blend_input_tensor, len(source.tensor))
        blend_tensor = self.gen_b(blend_input_tensor_pyd)

        return tensor2bgr(blend_tensor)
