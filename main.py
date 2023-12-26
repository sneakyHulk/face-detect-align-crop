import math
from typing import Tuple
from typing import Union

import tkinter as tk
import PIL.Image
import PIL.ImageTk
import numpy as np
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file('image.jpg')

# STEP 4: Detect faces in the input image.
detection_result = detector.detect(image)

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


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
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualize(
        image,
        detection_result
) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box

        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

        def angle_between_2_points(x1, y1, x2, y2):
            tan = (y2 - y1) / (x2 - x1)
            return np.degrees(np.arctan(tan))

        x_left_eye, y_left_eye = _normalized_to_pixel_coordinates(detection.keypoints[0].x, detection.keypoints[0].y,
                                                                  width, height)

        x_right_eye, y_right_eye = _normalized_to_pixel_coordinates(detection.keypoints[1].x, detection.keypoints[1].y,
                                                                    width, height)

        angle = angle_between_2_points(x_left_eye, y_left_eye, x_right_eye, y_right_eye)

        def center(x1, y1, x2, y2):
            return (x1 + x2) // 2, (y1 + y2) // 2

        xc, yc = center(x_left_eye, y_left_eye, x_right_eye, y_right_eye)
        dsize = max(width, height) * 2
        translation_matrix = np.array([
            [1, 0, dsize // 2 - xc],
            [0, 1, dsize // 2 - yc]
        ], dtype=np.float32)

        def distance(x1, y1, x2, y2):
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        eye_width = int(distance(x_left_eye, y_left_eye, x_right_eye, y_right_eye))
        top = int(dsize / 2 - 2 * eye_width)
        bottom = int(dsize / 2 + 2 * eye_width)

        def warpAffinePoint(x, y, m):
            return int(m[0][0] * x + m[0][1] * y + m[0][2]), int(m[1][0] * x + m[1][1] * y + m[1][2])

        translated_img = cv2.warpAffine(annotated_image, translation_matrix, (dsize, dsize), flags=cv2.INTER_CUBIC,
                                        borderValue=(255, 255, 255), borderMode=cv2.BORDER_CONSTANT)

        color, thickness, radius = (0, 255, 0), 2, 2
        kjeky = warpAffinePoint(x_left_eye, y_left_eye, translation_matrix)
        cv2.circle(translated_img, kjeky, thickness, color,
                   radius)

        rotation_matrix = cv2.getRotationMatrix2D((dsize // 2, dsize // 2), angle, 1)
        rotated_img = cv2.warpAffine(translated_img, rotation_matrix, (dsize, dsize), flags=cv2.INTER_CUBIC,
                                     borderValue=(255, 255, 255))

        an = annotated_image[bbox.origin_y:bbox.origin_y + bbox.width, bbox.origin_x:bbox.origin_x + bbox.height]
        rotated_cropped_img = rotated_img[top:bottom, top:bottom]
        an = img_return = cv2.resize(rotated_cropped_img, dsize=(600, 600), interpolation=cv2.INTER_CUBIC)

        an = cv2.cvtColor(an, cv2.COLOR_BGR2RGB)
        cv2.imshow('lol', an)
        cv2.waitKey(0)

        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                           width, height)
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image


def cv2toTkinter(_cv_img: np.ndarray):
    return PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))


# Create a window
window = tk.Tk()
window.title("OpenCV and Tkinter")

# Load an image using OpenCV
cv_img = cv2.cvtColor(cv2.imread("image.jpg"), cv2.COLOR_BGR2RGB)

# Get the image dimensions (OpenCV stores image data as NumPy ndarray)
height, width, _ = cv_img.shape

window.columnconfigure(0, weight=1)
window.columnconfigure(1, weight=1)
window.rowconfigure(0, weight=1)
window.rowconfigure(1)

frame = tk.Frame(master=window)
frame.grid(row=0, column=0, sticky="NESW")

# Create a canvas that can fit the above image
canvas = tk.Canvas(frame, scrollregion=(0, 0, width, height))
horizontal_bar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
horizontal_bar.pack(side=tk.BOTTOM, fill=tk.X)
horizontal_bar.config(command=canvas.xview)
vertical_bar = tk.Scrollbar(frame, orient=tk.VERTICAL)
vertical_bar.pack(side=tk.RIGHT, fill=tk.Y)
vertical_bar.config(command=canvas.yview)
canvas.config(xscrollcommand=horizontal_bar.set, yscrollcommand=vertical_bar.set)
canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

# Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))

# Add a PhotoImage to the Canvas
canvas.create_image(0, 0, image=photo, anchor=tk.NW)
button = tk.Button(window, text="click")
button.grid(row=1, column=0, sticky="NESW")

# Run the window loop
window.mainloop()

# STEP 5: Process the detection result. In this case, visualize it.
#image_copy = np.copy(image.numpy_view())
#annotated_image = visualize(image_copy, detection_result)
#rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
#cv2.imshow('lol', rgb_annotated_image)
#cv2.waitKey(0)