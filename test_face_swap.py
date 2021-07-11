from face_alignment import FaceAlignment, LandmarksType
from threaded_cv2.video_reader import VideoCapture
from tqdm import tqdm
import cv2
import torch

from afy.face_swap import Faceswap, SwapMethod

aligner = FaceAlignment(
    LandmarksType._2D,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    face_detector='blazeface',
)
get_face_swap = lambda **kwargs: Faceswap(aligner, SwapMethod.POISSON, **kwargs)
swapper = get_face_swap()

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def annotate_bboxes(im, bboxes):
    im = im.copy()
    for _, box in enumerate(bboxes):
        pos_1, pos_2 = (box[0], box[1]), (box[2], box[3])
        cv2.rectangle(im, pos_1, pos_2, color=(255, 0, 0))
    return im

def annotate_img():
    img_1 = cv2.imread('./avatars/opened_eyes.jpg')
    bboxes = swapper.get_bboxes(img_1)
    landmarks = swapper.get_landmarks(img_1)

    annotated_img = annotate_bboxes(img_1, bboxes)
    annotated_img = annotate_landmarks(img_1, landmarks)

    cv2.imwrite('68_landmarks_mtcnn.jpg', annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def swap_imgs():
    img_1 = cv2.imread('./avatars/opened_eyes.jpg')
    img_2 = cv2.imread('./avatars/closed_eyes.jpg')

    swapped = swapper.faceswap(img_1, img_2)
    cv2.imwrite(f'swapped_face_2.jpg', swapped)

def tune():
    img_1 = cv2.imread('./avatars/opened_eyes.jpg')
    img_2 = cv2.imread('./avatars/closed_eyes.jpg')

    for blur in tqdm(range(50, 100, 5)):
        swapper = get_face_swap(
            blur=blur / 100,
        )
        swapped = swapper.faceswap(img_1, img_2)
        cv2.imwrite(f'tuning/{blur}.jpg', swapped)

def frame_iter(capture: VideoCapture, description: str):
    return tqdm(
        capture,
        desc=description,
        total=capture.end_frame,
    )

def swap_video():
    img_2 = cv2.imread('./avatars/closed_eyes.jpg')
    with VideoCapture('./test-video.mp4') as cap:
        for idx, frame in enumerate(frame_iter(cap, 'Face-swapping video')):
            swapped = swapper.faceswap(frame, img_2)
            cv2.imwrite(f'results/{idx:08}.jpg', swapped)

def swap_video_same():
    with VideoCapture('./test-video.mp4') as cap:
        for idx, frame in enumerate(frame_iter(cap, 'Face-swapping video')):
            swapped = swapper.faceswap(frame, frame)
            cv2.imwrite(f'results/{idx:08}.jpg', swapped)

def main():
    swap_video_same()

if __name__ == "__main__":
    main()
