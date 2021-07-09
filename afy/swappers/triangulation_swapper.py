"""
Inspired from https://github.com/oriolbernal/face-swap/blob/master/face_swap.py
"""
import cv2
import numpy as np

from afy.custom_typings import CV2Image
from afy.swappers import Swapper

def get_index_pt(point: tuple, points: np.ndarray):
    array = np.where((points == point).all(axis=1))
    return array[0][0]

def get_delaunay_triangulation(
    landmarks_points: np.ndarray,
    convexhull,
):
    '''
    Gets delaunay triangulation of the given landmarks points in the given convex
    hull
    '''
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    points = []
    for point in landmarks_points:
        temp = tuple(point)
        points.append(temp)
        subdiv.insert(temp)
    points = np.array(points)
    triangles = np.array(subdiv.getTriangleList(), dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        index_pt1 = get_index_pt((t[0], t[1]), points)
        if index_pt1 is None:
            continue
        index_pt2 = get_index_pt((t[2], t[3]), points)
        if index_pt2 is None:
            continue
        index_pt3 = get_index_pt((t[4], t[5]), points)
        if index_pt3 is None:
            continue
        indexes_triangles.append([index_pt1, index_pt2, index_pt3])

    return indexes_triangles

def get_new_face(
    img: CV2Image,
    img2: CV2Image,
    landmarks_points: np.ndarray,
    landmarks_points2: np.ndarray,
    indexes_triangles: list,
):
    img_new_face = np.zeros(img2.shape, np.uint8)

    # Triangulation of both faces
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)

        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                           [tr1_pt2[0] - x, tr1_pt2[1] - y],
                           [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

        # Triangulation of second face
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        # Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # Reconstructing destination face
        img_new_face_rect_area = img_new_face[y: y + h, x: x + w]
        img_new_face_rect_area_gray = cv2.cvtColor(img_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(
            img_new_face_rect_area_gray,
            1,
            255,
            cv2.THRESH_BINARY_INV
        )
        warped_triangle = cv2.bitwise_and(
            warped_triangle,
            warped_triangle,
            mask=mask_triangles_designed
        )

        img_new_face_rect_area = cv2.add(img_new_face_rect_area, warped_triangle)
        img_new_face[y: y + h, x: x + w] = img_new_face_rect_area

    return img_new_face

def change_face(img: CV2Image, convexhull, new_face: CV2Image):
    img_face_mask = np.zeros_like(img[:, :, 0])
    img_head_mask = cv2.fillConvexPoly(img_face_mask, convexhull, 255)
    img_face_mask = cv2.bitwise_not(img_head_mask)

    img2_head_noface = cv2.bitwise_and(img, img, mask=img_face_mask)
    result = cv2.add(img2_head_noface, new_face)

    (x, y, w, h) = cv2.boundingRect(convexhull)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone = cv2.seamlessClone(result, img, img_head_mask, center_face2, cv2.NORMAL_CLONE)

    return seamlessclone

class TriangulationSwapper(Swapper):
    def swap_imgs(
        self,
        im1: CV2Image,
        im2: CV2Image,
        landmarks1: np.ndarray,
        landmarks2: np.ndarray,
    ) -> CV2Image:
        '''Swap the images using their corresponding landmarks'''
        convexhull = cv2.convexHull(landmarks1)
        convexhull2 = cv2.convexHull(landmarks2)
        indexes_triangles2 = get_delaunay_triangulation(landmarks2, convexhull2)

        img_new_face = get_new_face(im2, im1, landmarks2, landmarks1, indexes_triangles2)
        img_changed_face = change_face(im1, convexhull, img_new_face)

        return img_changed_face
