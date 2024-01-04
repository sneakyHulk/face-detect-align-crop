import PIL.Image
import PIL.ImageOps
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import dlib
from typing import Tuple
from typing import Union
import math
import cv2


# TODO: cvlib


def use_cnn(image: PIL.Image.Image):
    if not hasattr(use_cnn, "detector"):
        # print("New cnn detector!")
        use_cnn.detector = detector_cnn = dlib.cnn_face_detection_model_v1("data/mmod_human_face_detector.dat")

    detection_result = use_cnn.detector(np.array(image), 1)

    for detection in detection_result:
        bbox = detection.rect

        yield bbox.left(), bbox.top(), bbox.right(), bbox.bottom()


def use_hog(image: PIL.Image.Image):
    if not hasattr(use_hog, "detector"):
        # print("New hog detector!")
        use_hog.detector = dlib.get_frontal_face_detector()

    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    detection_result = use_hog.detector(gray, 1)

    for bbox in detection_result:
        yield bbox.left(), bbox.top(), bbox.right(), bbox.bottom()


def use_mediapipe(image: PIL.Image.Image):
    if not hasattr(use_mediapipe, "detector"):
        # print("New mediapipe detector!")
        use_mediapipe.detector = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(base_options=python.BaseOptions(model_asset_path='data/detector.tflite')))

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image))
    detection_result = use_mediapipe.detector.detect(mp_image)

    for detection in detection_result.detections:
        bbox = detection.bounding_box

        def _normalized_to_pixel_coordinates(
                normalized_x: float, normalized_y: float, image_width: int,
                image_height: int) -> Union[None, Tuple[int, int]]:
            """Converts normalized value pair to pixel coordinates."""

            # Checks if the float value is between 0 and 1.
            def is_valid_normalized_value(value: float) -> bool:
                return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                                  math.isclose(1, value))

            if not (is_valid_normalized_value(normalized_x) and
                    is_valid_normalized_value(normalized_y)):
                # TODO: Draw coordinates even if it's outside of the image bounds.
                return None
            x_px: int = min(math.floor(normalized_x * image_width), image_width - 1)
            y_px: int = min(math.floor(normalized_y * image_height), image_height - 1)
            return x_px, y_px

        x_eye_left, y_eye_left = _normalized_to_pixel_coordinates(detection.keypoints[0].x,
                                                                  detection.keypoints[0].y,
                                                                  image.width, image.height)

        x_eye_right, y_eye_right = _normalized_to_pixel_coordinates(detection.keypoints[1].x,
                                                                    detection.keypoints[1].y,
                                                                    image.width, image.height)

        yield bbox.origin_x, bbox.origin_y, bbox.origin_x + bbox.width + 1, bbox.origin_y + bbox.height + 1, x_eye_left, y_eye_left, x_eye_right, y_eye_right


def face_align_crop(image: PIL.Image.Image, output_width, output_height, x_start_bbox, y_start_bbox, x_end_bbox,
                    y_end_bbox, x_eye_left=None, y_eye_left=None, x_eye_right=None,
                    y_eye_right=None) -> PIL.Image.Image:
    def angle_between_2_points(x1, y1, x2, y2):
        tan = (y2 - y1) / (x2 - x1)
        return np.degrees(np.arctan(tan))

    def center(x1, y1, x2, y2):
        return (x1 + x2) // 2, (y1 + y2) // 2

    def warpAffinePoint(x, y, m):
        return int(m[0][0] * x + m[0][1] * y + m[0][2]), int(m[1][0] * x + m[1][1] * y + m[1][2])

    def cv2toPILRotationMatrix(matrix):
        return np.linalg.inv(np.concatenate((matrix, np.array([[0, 0, 1]], dtype=np.float32)), axis=0)).flatten()[:6]

    if x_eye_left and y_eye_left and x_eye_right and y_eye_right:
        xc, yc = center(x_eye_left, y_eye_left, x_eye_right, y_eye_right)
    else:
        xc, yc = center(x_start_bbox, y_start_bbox, x_end_bbox, y_end_bbox)

    dsize = max(image.width, image.height) * 2
    translation_matrix = np.array([
        [1, 0, dsize // 2 - xc],
        [0, 1, dsize // 2 - yc],
    ], dtype=np.float32)

    image = image.transform((dsize, dsize), PIL.Image.Transform.AFFINE, cv2toPILRotationMatrix(translation_matrix),
                            PIL.Image.BICUBIC, fillcolor='white')
    xc, yc = warpAffinePoint(xc, yc, translation_matrix)

    if x_eye_left and y_eye_left and x_eye_right and y_eye_right:
        angle = angle_between_2_points(x_eye_left, y_eye_left, x_eye_right, y_eye_right)
        rotation_matrix = cv2.getRotationMatrix2D((dsize // 2, dsize // 2), angle, 1)

        image = image.transform((dsize, dsize), PIL.Image.Transform.AFFINE, cv2toPILRotationMatrix(rotation_matrix),
                                PIL.Image.BICUBIC, fillcolor='white')
        xc, yc = warpAffinePoint(xc, yc, rotation_matrix)

    w = output_width / abs(x_end_bbox - x_start_bbox) * 0.5
    v = output_height / abs(y_end_bbox - y_start_bbox) * 0.5

    if w < v:
        image = PIL.ImageOps.scale(image, w)
        xc, yc = xc * w, yc * w
    else:
        image = PIL.ImageOps.scale(image, v)
        xc, yc = xc * v, yc * v
    image = image.crop((xc - output_width // 2, yc - output_height // 2,
                        xc + output_width // 2, yc + output_height // 2))

    return image
