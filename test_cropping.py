from face_alignment import FaceAlignment, LandmarksType
from PIL import Image
import cv2
import numpy as np
import torch

def extract_face(image: Image.Image, box) -> Image.Image:
    margin = 25
    if isinstance(box, np.ndarray):
        box = box[:4]
    box[0] -= margin
    box[1] -= margin
    box[2] += margin
    box[3] += margin
    return image.crop(box).resize((256, 256))

@torch.no_grad()
def get_face(image_numpy, aligner: FaceAlignment):
    color_img = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
    bbox = aligner.face_detector.detect_from_image(color_img.copy())[0]
    pil_image = Image.fromarray(color_img)
    return extract_face(pil_image, bbox)

def main():
    aligner = FaceAlignment(
        LandmarksType._2D,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        face_detector='blazeface',
    )
    img_1 = cv2.imread('./avatars/opened_eyes.jpg')
    face = get_face(img_1, aligner)
    face.show()

if __name__ == "__main__":
    main()
