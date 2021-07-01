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
See the above for an explanation of the code below.
To run the script you'll need to install dlib (http://dlib.net) including its
Python bindings, and OpenCV. You'll also need to obtain the trained model from
sourceforge:
    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
"""
import cv2
import dlib
import numpy

BLUR_AMOUNT = 0.6
FEATHER_AMOUNT = 11
SCALE_FACTOR = 1

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

EYES_BROWS_POINTS = (
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS
)

NOSE_MOUTH_POINTS = NOSE_POINTS + MOUTH_POINTS

# Points used to line up the images.
ALIGN_POINTS = (
    LEFT_BROW_POINTS +
    RIGHT_EYE_POINTS +
    LEFT_EYE_POINTS +
    RIGHT_BROW_POINTS +
    NOSE_POINTS +
    MOUTH_POINTS
)

def draw_convex_hull(img, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(img, points, color=color)

def warp_im(img, M, dshape):
    output_im = numpy.zeros(dshape, dtype=img.dtype)
    cv2.warpAffine(img,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def prepare_transformation_points(points: numpy.ndarray):
    points = points.astype(numpy.float64)
    mean = numpy.mean(points, axis=0)
    points -= mean
    std = numpy.std(points)
    points /= std
    return points, mean, std

def transformation_from_points(points1, points2):
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

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])
class Faceswap:
    def __init__(self,
        predictor_path = './shape_predictor_68_face_landmarks.dat',
        overlay_eyesbrows = True,
        overlay_nosemouth = True,
        only_mouth = False,
        feather = FEATHER_AMOUNT,
        blur = BLUR_AMOUNT,
        ignore_nofaces = False,
        colour_correct = True
    ):
        self.predictor_path = predictor_path
        self.blur = blur
        self.detector = dlib.get_frontal_face_detector()
        self.feather = feather
        self.predictor = dlib.shape_predictor(self.predictor_path)
        self.overlay_points = []
        self.landmark_hashes = {}
        self.ignore_nofaces = ignore_nofaces
        self.colour_correct = colour_correct

        # TODO: this should be a little bit less messy
        if only_mouth:
            self.overlay_points.append(MOUTH_POINTS)
        else:
            if overlay_eyesbrows:
                self.overlay_points.append(EYES_BROWS_POINTS)

            if overlay_nosemouth:
                self.overlay_points.append(NOSE_MOUTH_POINTS)

    def _correct_colours(self, im1, im2, landmarks1):
        if not self.colour_correct:
            return im2
        blur_amount = self.blur * numpy.linalg.norm(
                                  numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                                  numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

        # Avoid divide-by-zero errors.
        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

        return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                            im2_blur.astype(numpy.float64))

    def _get_face_mask(self, img, landmarks):
        img = numpy.zeros(img.shape[:2], dtype=numpy.float64)

        for group in self.overlay_points:
            draw_convex_hull(img, landmarks[group], color=1)

        img = numpy.array([img, img, img]).transpose((1, 2, 0))

        img = (cv2.GaussianBlur(img, (self.feather, self.feather), 0) > 0) * 1.0
        img = cv2.GaussianBlur(img, (self.feather, self.feather), 0)

        return img

    def _get_landmarks(self, img):
        # This is by far the slowest part of the whole algorithm, so we
        # cache the landmarks if the image is the same, especially when
        # dealing with videos this makes things twice as fast
        img_hash = str(abs(hash(img.data.tobytes())))

        if img_hash in self.landmark_hashes:
            return self.landmark_hashes[img_hash]

        rects = self.detector(img, 1)

        if len(rects) != 1:
            raise ValueError('There should be one face in the image')

        landmarks = numpy.matrix([
            [p.x, p.y] for p in self.predictor(img, rects[0]).parts()
        ])

        # Save to image cache
        self.landmark_hashes[img_hash] = landmarks

        return landmarks

    def faceswap(self, head, face):
        '''Inserts the face into the head'''
        im1 = head
        im2 = face

        try:
            landmarks1 = self._get_landmarks(head)
            landmarks2 = self._get_landmarks(face)
        except ValueError:
            return im1

        output_im = im1

        M = transformation_from_points(
            landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS]
        )

        mask = self._get_face_mask(im2, landmarks2)
        warped_mask = warp_im(mask, M, im1.shape)

        combined_mask = numpy.max(
            [self._get_face_mask(im1, landmarks1), warped_mask],
            axis=0
        )

        warped_im2 = warp_im(im2, M, im1.shape)

        warped_corrected_im2 = self._correct_colours(
            im1, warped_im2, landmarks1
        )

        output_im = output_im * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

        return output_im.astype(numpy.uint8)
