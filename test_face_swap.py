import cv2
# from tqdm import tqdm

from afy.face_swap_2 import Faceswap

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

def main():
    img_1 = cv2.imread('./avatars/opened_eyes.jpg')
    img_2 = cv2.imread('./avatars/closed_eyes.jpg')

    swapper = Faceswap()

    swapped = swapper.faceswap(img_1, img_2)
    cv2.imwrite('swapped_face_2.jpg', swapped)

if __name__ == "__main__":
    main()
