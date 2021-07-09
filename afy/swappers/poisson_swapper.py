'''Obtained from: https://github.com/wuhuikai/FaceSwap/blob/master/face_swap.py'''
from dataclasses import dataclass
import logging

import cv2
import numpy as np
from scipy import spatial

from afy.custom_typings import CV2Image
from afy.swappers import Swapper
from afy.swappers.eds_swapper import get_im_blur

## 3D Transform
def bilinear_interpolate(img, coords: np.ndarray):
    """
    Interpolates over every image channel
    http://en.wikipedia.org/wiki/Bilinear_interpolation

    :param img: max 3 channel image
    :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
    :returns: array of interpolated pixels with same shape as coords
    """
    int_coords = coords.astype(np.int32)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    # 4 Neighour pixels
    q11 = img[y0, x0]
    q21 = img[y0, x0 + 1]
    q12 = img[y0 + 1, x0]
    q22 = img[y0 + 1, x0 + 1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T

def grid_coordinates(points):
    """
    x,y grid coordinates within the ROI of supplied points

    :param points: points to generate grid coordinates
    :returns: array of (x, y) coordinates
    """
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0]) + 1
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1]) + 1

    return np.asarray([(x, y) for y in range(ymin, ymax)
                       for x in range(xmin, xmax)], np.uint32)

def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    """
    Warp each triangle from the src_image only within the ROI of the
    destination image (points in dst_points).
    """
    roi_coords = grid_coordinates(dst_points)
    # indices to vertices. -1 if pixel is not in any triangle
    roi_tri_indices = delaunay.find_simplex(roi_coords)

    for simplex_index in range(len(delaunay.simplices)):
        coords = roi_coords[roi_tri_indices == simplex_index]
        num_coords = len(coords)
        out_coords = np.dot(tri_affines[simplex_index],
                            np.vstack((coords.T, np.ones(num_coords))))
        x, y = coords.T
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)

    return None

def triangular_affine_matrices(vertices, src_points, dst_points):
    """
    Calculate the affine transformation matrix for each triangle (x,y) vertex
    from dst_points to src_points

    :param vertices: array of triplet indices to corners of triangle
    :param src_points: array of [x, y] points to landmarks for source image
    :param dst_points: array of [x, y] points to landmarks for destination image
    :returns: 2 x 3 affine matrix transformation for a triangle
    """
    ones = [1, 1, 1]
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dst_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat

def warp_image_3d(src_img, src_points, dst_points, dst_shape, dtype=np.uint8):
    rows, cols = dst_shape[:2]
    result_img = np.zeros((rows, cols, 3), dtype=dtype)

    delaunay = spatial.Delaunay(dst_points)
    tri_affines = np.asarray(list(triangular_affine_matrices(
        delaunay.simplices, src_points, dst_points)))

    process_warp(src_img, result_img, tri_affines, dst_points, delaunay)

    return result_img

def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(np.dot(points1.T, points2))
    R = (np.dot(U, Vt)).T

    return np.vstack([np.hstack([s2 / s1 * R,
                                (c2.T - np.dot(s2 / s1 * R, c1.T))[:, np.newaxis]]),
                      np.array([[0., 0., 1.]])])

def warp_image_2d(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)

    return output_im

def apply_mask(img, mask):
    """
    Apply mask to supplied image

    :param img: max 3 channel image
    :param mask: [0-255] values in mask
    :returns: new image with mask applied
    """
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return masked_img

def get_output(mask, warped_src_face, dst_face, dst_shape, dst_img):
    ## Poisson Blending
    r = cv2.boundingRect(mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)

    x, y, w, h = dst_shape
    dst_img_cp = dst_img.copy()
    dst_img_cp[y:y + h, x:x + w] = output

    return dst_img_cp

def parse_data(im, points, r=10):
    im_w, im_h = im.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)

    x, y = max(0, left - r), max(0, top - r)
    w, h = min(right + r, im_h) - x, min(bottom + r, im_w) - y

    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y + h, x:x + w]

@dataclass
class PoissonSwapper(Swapper):
    blur = 0.75
    correct_color = True
    end = 48
    radius = 10
    warp_2d = True

    def _mask_from_points(self, size, points, erode_flag=1):
        kernel = np.ones((self.radius, self.radius), np.uint8)

        mask = np.zeros(size, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
        if erode_flag:
            mask = cv2.erode(mask, kernel,iterations=1)

        return mask

    def _get_mask(self, size, dst_points, warped_src_face):
        h, w = size
        mask = self._mask_from_points((h, w), dst_points)
        mask_src = np.mean(warped_src_face, axis=2) > 0
        return np.asarray(mask * mask_src, dtype=np.uint8)

    def _correct_colours(self, im1, im2, landmarks1):
        im1_blur, im2_blur = get_im_blur(
            im1, im2, landmarks1, self.blur
        )

        result = im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def _correct_warped_face_color(self, warped_src_face, mask, dst_face, dst_points):
        warped_src_face = apply_mask(warped_src_face, mask)
        dst_face_masked = apply_mask(dst_face, mask)
        return self._correct_colours(dst_face_masked, warped_src_face, dst_points)

    def _get_2d_warped_mask(
        self,
        warped_src_face,
        dst_points,
        src_points,
        src_face,
        size
    ):
        h, w = size
        unwarped_src_face = warp_image_3d(
            warped_src_face,
            dst_points[:self.end],
            src_points[:self.end],
            src_face.shape[:2]
        )
        warped_src_face = warp_image_2d(
            unwarped_src_face,
            transformation_from_points(dst_points, src_points),
            (h, w, 3)
        )

        mask = self._mask_from_points((h, w), dst_points)
        mask_src = np.mean(warped_src_face, axis=2) > 0
        mask = np.asarray(mask * mask_src, dtype=np.uint8)
        return mask

    def _face_swap(self, src_face, dst_face, src_points, dst_points, dst_shape, dst_img):
        dst_size = tuple(dst_face.shape[:2])

        ## 3d warp
        warped_src_face = warp_image_3d(
            src_face, src_points[:self.end], dst_points[:self.end], dst_size
        )

        ## Mask for blending
        mask = self._get_mask(dst_size, dst_points, warped_src_face)

        ## Correct color
        if self.correct_color:
            warped_src_face = self._correct_warped_face_color(
                warped_src_face, mask, dst_face, dst_points
            )

        ## 2D warp
        if self.warp_2d:
            mask = self._get_2d_warped_mask(
                warped_src_face, dst_points, src_points, src_face, dst_size
            )

        ## Shrink the mask
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        return get_output(mask, warped_src_face, dst_face, dst_shape, dst_img)

    def swap_imgs(
        self,
        im1: CV2Image,
        im2: CV2Image,
        landmarks1: np.ndarray,
        landmarks2: np.ndarray,
    ) -> CV2Image:
        dst_points, dst_shape, dst_face = parse_data(im1, landmarks1)
        src_points, _, src_face = parse_data(im2, landmarks2)

        return self._face_swap(src_face, dst_face, src_points, dst_points, dst_shape, im1)
