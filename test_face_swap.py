import cv2
from tqdm import tqdm

from afy.face_swap_2 import Faceswap

def main():
    img_1 = cv2.imread('./avatars/opened_eyes.jpg')
    img_2 = cv2.imread('./avatars/closed_eyes.jpg')

    swapper = Faceswap()

    swapped = swapper.faceswap(img_1, img_2)
    print(swapped.dtype, img_1.dtype)
    cv2.imshow('img_1', img_1)
    cv2.imshow('img_2', img_2)
    cv2.imshow('swapped_face', swapped)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
