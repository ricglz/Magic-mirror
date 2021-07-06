"""
Faceswap library
Based on http://matthewearl.github.io/2015/07/28/switching-eds-with-python/
Copyright (c) 2015 Matthew Earl
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
    NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
    USE OR OTHER DEALINGS IN THE SOFTWARE.
This is the code behind the Switching Eds blog post:
    http://matthewearl.github.io/2015/07/28/switching-eds-with-python/
"""
import numpy as np
import cv2

from afy.custom_typings import CV2Image

current_blur = 0
current_feather = 0

JAW_POINTS = list(range(0, 17))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_POINTS = list(range(48, 68))

EYES_BROWS_POINTS = (
    RIGHT_BROW_POINTS +
    LEFT_BROW_POINTS +
    RIGHT_EYE_POINTS +
    LEFT_EYE_POINTS
)

NOSE_MOUTH_POINTS = NOSE_POINTS + MOUTH_POINTS

# Points used to line up the images.
ALIGN_POINTS = JAW_POINTS + EYES_BROWS_POINTS + NOSE_MOUTH_POINTS

overlay_points = [EYES_BROWS_POINTS, NOSE_MOUTH_POINTS]

def prepare_transformation_points(points: np.ndarray):
    points = points.astype(np.float64)
    mean = np.mean(points, axis=0)
    points -= mean
    std = np.std(points)
    points /= std
    return points, mean, std

def transformation_from_points(points1, points2) -> np.ndarray:
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1, c1, s1 = prepare_transformation_points(points1)
    points2, c2, s2 = prepare_transformation_points(points2)

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    s_divide_r = (s2 / s1) * R
    return np.vstack([np.hstack((s_divide_r, c2.T - s_divide_r * c1.T)),
                         np.matrix([0., 0., 1.])])

def warp_im(img, M, dshape):
    output_im = np.zeros(dshape, dtype=img.dtype)
    cv2.warpAffine(img,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def draw_convex_hull(img, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(img, points, color=color)

def get_face_mask(img, landmarks):
    img = np.zeros(img.shape[:2], dtype=np.float64)
    feather = current_feather

    for group in overlay_points:
        draw_convex_hull(img, landmarks[group], color=1)

    img = np.array([img, img, img]).transpose((1, 2, 0))

    img = (cv2.GaussianBlur(img, (feather, feather), 0) > 0) * 1.0
    img = cv2.GaussianBlur(img, (feather, feather), 0)

    return img

def get_combined_mask(
    im1: CV2Image,
    landmarks1: np.matrix,
    im2: CV2Image,
    landmarks2: np.matrix,
    M,
):
    warped_mask = warp_im(
        get_face_mask(im2, landmarks2),
        M,
        im1.shape
    )

    return np.max(
        [get_face_mask(im1, landmarks1), warped_mask],
        axis=0
    )

def correct_colours(im1, im2, landmarks1):
    '''Correct the color of the new face based on the blur quantity'''
    blur = current_blur
    blur_amount = blur * np.linalg.norm(
                         np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                         np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
                                     im2_blur.astype(np.float64))

def swap_imgs(
    im1: CV2Image,
    im2: CV2Image,
    landmarks1: np.ndarray,
    landmarks2: np.ndarray,
) -> CV2Image:
    '''Swap the images using their corresponding landmarks'''
    output_im = im1
    landmarks1 = np.matrix(landmarks1)
    landmarks2 = np.matrix(landmarks2)

    M = transformation_from_points(
        landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS]
    )

    combined_mask = get_combined_mask(
        im1, landmarks1, im2, landmarks2, M
    )

    warped_im2 = warp_im(im2, M, im1.shape)

    warped_corrected_im2 = correct_colours(
        im1, warped_im2, landmarks1
    )

    return output_im * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

def get_swap_function(blur: int, feather: int):
    global current_blur, current_feather

    current_blur, current_feather = blur, feather

    return swap_imgs
