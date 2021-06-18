#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inspired from https://github.com/oriolbernal/face-swap/blob/master/face_swap.py
"""
from typing import Tuple

import cv2
import numpy as np

Face_Data = Tuple[np.ndarray, np.ndarray]

def extract_index(array: np.ndarray):
    '''Extracts index'''
    index = None
    for num in array[0]:
        index = num
        break
    return index

def get_delaunay_triangulation(
    landmarks_points: np.ndarray,
    convexhull: cv2.convexHull,
):
    '''
    Gets delaunay triangulation of the given landmarks points
    in the given convex hull
    '''
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    points = []
    for point in landmarks_points:
        temp = tuple(point)
        points.append(temp)
        subdiv.insert(temp)
    points = np.array(points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    return indexes_triangles

def get_new_face(
    img: np.ndarray,
    img2: np.ndarray,
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
        _, mask_triangles_designed = cv2.threshold(img_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img_new_face_rect_area = cv2.add(img_new_face_rect_area, warped_triangle)
        img_new_face[y: y + h, x: x + w] = img_new_face_rect_area

    return img_new_face


def change_face(img, convexhull, new_face):
    img_face_mask = np.zeros_like(img[:, :, 0])
    img_head_mask = cv2.fillConvexPoly(img_face_mask, convexhull, 255)
    img_face_mask = cv2.bitwise_not(img_head_mask)

    img2_head_noface = cv2.bitwise_and(img, img, mask=img_face_mask)
    result = cv2.add(img2_head_noface, new_face)

    (x, y, w, h) = cv2.boundingRect(convexhull)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone = cv2.seamlessClone(result, img, img_head_mask, center_face2, cv2.NORMAL_CLONE)

    return seamlessclone


def swap_faces(face_data: Face_Data, face_data_2: Face_Data):
    img, landmarks_points = face_data
    img2, landmarks_points2 = face_data_2

    convexhull = cv2.convexHull(landmarks_points)
    convexhull2 = cv2.convexHull(landmarks_points2)
    indexes_triangles2 = get_delaunay_triangulation(landmarks_points2, convexhull2)

    img_new_face = get_new_face(img2, img, landmarks_points2, landmarks_points, indexes_triangles2)
    img_changed_face = change_face(img, convexhull, img_new_face)

    return img_changed_face
