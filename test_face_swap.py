import cv2
from tqdm import tqdm

from afy.face_swap_2 import Faceswap

swapper = Faceswap()

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
    # bboxes = swapper._get_bboxes(img_1)
    landmarks = swapper._get_landmarks(img_1)

    # annotated_img = annotate_bboxes(img_1, bboxes)
    annotated_img = annotate_landmarks(img_1, landmarks)

    cv2.imwrite('68_landmarks_mtcnn.jpg', annotated_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def swap_imgs():
    img_1 = cv2.imread('./avatars/swap_opened_eyes.jpg')
    img_2 = cv2.imread('./avatars/closed_eyes.jpg')

    swapped = swapper.faceswap(img_1, img_2)
    cv2.imwrite(f'swapped_face_2.jpg', swapped)

def tune_blur_feather_swap():
    img_1 = cv2.imread('./avatars/opened_eyes.jpg')
    img_2 = cv2.imread('./avatars/closed_eyes.jpg')

    max_blur = 100
    for idx, blur in tqdm(enumerate(range(15, max_blur))):
        # desc = f'Running {idx + 1}'
        feather = 35
        # for feather in tqdm(range(15, 37, 2), desc):
        swapper = Faceswap(blur=blur / 10, feather=feather)
        swapped = swapper.faceswap(img_1, img_2)
        cv2.imwrite(f'swapped_face_{blur}_{feather}.jpg', swapped)

def main():
    swap_imgs()

if __name__ == "__main__":
    main()
