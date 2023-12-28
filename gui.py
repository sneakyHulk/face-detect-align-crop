import tkinter as tk
from tkinter import filedialog as fd
import PIL.Image
import PIL.ImageTk
import PIL.ImageOps
import numpy as np
import os
import platform
import cv2
from pi_heif import register_heif_opener
from dataclasses import dataclass
from tkinter import ttk
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple
from typing import Union
import math

base_options = python.BaseOptions(model_asset_path='detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

register_heif_opener()


def pathtoTkinter(image_path):
    cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return cv2toTkinter(cv_img)


def cv2toTkinter(cv_img: np.ndarray):
    return PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))


class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.output_height = 600
        self.output_width = 600

        # Initialize root window
        self.wm_title("FACE-DETECT-ALIGN-CROP")
        # self.resizable(False, False)
        self.geometry('1280x734')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Initialize container frame which can be switched
        container = tk.Frame(self)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for Frame in (SelectImagesPage, ProcessingPages):
            self.frames[Frame] = Frame(container, self)
            self.frames[Frame].grid(row=0, column=0, sticky="nsew")

        self.show_frame(SelectImagesPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        # raises the current frame to the top
        frame.tkraise()
        return self.frames[cont]


@dataclass
class CanvasImageSelectedContainer:
    canvas: tk.Canvas
    image: PIL.ImageTk.PhotoImage
    selected: bool


def face_align_crop(image: PIL.Image.Image, x_start_bbox, y_start_bbox, x_end_bbox, y_end_bbox,
                    output_width=600, output_height=600, x_eye_left=None, y_eye_left=None, x_eye_right=None,
                    y_eye_right=None):
    def angle_between_2_points(x1, y1, x2, y2):
        tan = (y2 - y1) / (x2 - x1)
        return np.degrees(np.arctan(tan))

    def center(x1, y1, x2, y2):
        return (x1 + x2) // 2, (y1 + y2) // 2

    def distance(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

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

    # image = np.array(image)
    # image = cv2.warpAffine(image, translation_matrix, (dsize, dsize), flags=cv2.INTER_CUBIC,
    #                       borderValue=(255, 255, 255), borderMode=cv2.BORDER_CONSTANT)
    image = image.transform((dsize, dsize), PIL.Image.Transform.AFFINE, cv2toPILRotationMatrix(translation_matrix),
                            PIL.Image.BICUBIC)
    xc, yc = warpAffinePoint(xc, yc, translation_matrix)

    if x_eye_left and y_eye_left and x_eye_right and y_eye_right:
        angle = angle_between_2_points(x_eye_left, y_eye_left, x_eye_right, y_eye_right)
        rotation_matrix = cv2.getRotationMatrix2D((dsize // 2, dsize // 2), angle, 1)

        # image = cv2.warpAffine(image, rotation_matrix, (dsize, dsize), flags=cv2.INTER_CUBIC,
        #                       borderValue=(255, 255, 255))
        image = image.transform((dsize, dsize), PIL.Image.Transform.AFFINE, cv2toPILRotationMatrix(rotation_matrix),
                                PIL.Image.BICUBIC)

        xc, yc = warpAffinePoint(xc, yc, rotation_matrix)

    w = output_width / abs(x_end_bbox - x_start_bbox) * 0.5
    v = output_height / abs(y_end_bbox - y_start_bbox) * 0.5

    # image = PIL.Image.fromarray(image)
    if w < v:
        image = PIL.ImageOps.scale(image, w)
        xc, yc = xc * w, yc * w
    else:
        image = PIL.ImageOps.scale(image, v)
        xc, yc = xc * v, yc * v
    image = image.crop((xc - output_width // 2, yc - output_height // 2,
                        xc + output_width // 2, yc + output_height // 2))

    return image


class ProcessingPage(tk.Frame):

    def update_images(self):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.image)
        detection_result = detector.detect(image)

        for i, detection in enumerate(detection_result.detections):
            bbox = detection.bounding_box

            canvas = tk.Canvas(self.images_frame, width=self.controller.output_width - 2,
                               height=self.controller.output_height - 2, background='white', highlightthickness=1,
                               highlightbackground="black")

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

            height, width, _ = self.image.shape

            x_eye_left, y_eye_left = _normalized_to_pixel_coordinates(detection.keypoints[0].x,
                                                                      detection.keypoints[0].y,
                                                                      width, height)

            x_eye_right, y_eye_right = _normalized_to_pixel_coordinates(detection.keypoints[1].x,
                                                                        detection.keypoints[1].y,
                                                                        width, height)

            image = face_align_crop(PIL.Image.fromarray(self.image), bbox.origin_x, bbox.origin_y,
                                    bbox.origin_x + bbox.width + 1, bbox.origin_y + bbox.height + 1,
                                    self.controller.output_width, self.controller.output_height, x_eye_left, y_eye_left,
                                    x_eye_right, y_eye_right)

            # angle = angle_between_2_points(x_left_eye, y_left_eye, x_right_eye, y_right_eye)
            #
            # xc, yc = center(x_left_eye, y_left_eye, x_right_eye, y_right_eye)
            #
            # bbox = detection.bounding_box
            # xc2, yc2 = center(bbox.origin_x, bbox.origin_y, bbox.origin_x + bbox.width + 1,
            #                  bbox.origin_y + bbox.height + 1)
            #
            ## eye_width = int(distance(x_left_eye, y_left_eye, x_right_eye, y_right_eye))
            ## top = int(dsize / 2 - 2 * eye_width)
            ## bottom = int(dsize / 2 + 2 * eye_width)
            #
            # xc2, yc2 = warpAffinePoint(xc2, yc2, translation_matrix)
            #
            # xc2, yc2 = warpAffinePoint(xc2, yc2, rotation_matrix)
            #
            # cv2.circle(rotated_img, (xc2, yc2), thickness=2, color=(255, 0, 0), radius=2)
            #
            # w = self.controller.output_width / bbox.width * 0.5
            # v = self.controller.output_height / bbox.height * 0.5
            #
            # img = PIL.Image.fromarray(rotated_img)
            #
            ## img = PIL.ImageOps.fit(img, (self.controller.output_width, self.controller.output_height))
            img = PIL.ImageTk.PhotoImage(image)

            canvas.create_image(0, 0, image=img, anchor=tk.NW)

            def select_image(event):
                for element in self.data:
                    if event.widget == element.canvas:
                        if element.selected:
                            element.selected = False
                            element.canvas.configure(highlightbackground="black")
                        else:
                            element.selected = True
                            element.canvas.configure(highlightbackground="red")
                    else:
                        element.selected = False
                        element.canvas.configure(highlightbackground="black")

            canvas.bind("<Button-1>", select_image)

            canvas.grid(row=i, sticky="nsew", padx=1, pady=1)
            self.data.append(CanvasImageSelectedContainer(canvas, img, False))

        for detection in detection_result.detections:
            bbox = detection.bounding_box

            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(self.image, start_point, end_point, color=(255, 0, 0), thickness=1)

    def __init__(self, parent, controller, image_path):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.image_path = image_path
        self.image = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
        self.data: list[CanvasImageSelectedContainer] = []

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1, uniform="x")
        self.grid_columnconfigure(1, minsize=controller.output_width + 20)

        self.canvas = tk.Canvas(self, highlightthickness=1, highlightbackground="black")
        self.canvas.grid(row=0, column=0, padx=2, pady=2, sticky="nswe")
        self.canvas.bind("<Configure>", self.onCanvasConfigure)

        self.scroll_frame = ScrollFrame(self)
        self.images_frame = self.scroll_frame.viewPort
        self.scroll_frame.grid(row=0, column=1, sticky="nswe")

        self.update_images()

    def onCanvasConfigure(self, event):
        self.canvas.img = PIL.Image.fromarray(self.image)
        self.canvas.img = PIL.ImageOps.pad(self.canvas.img, (event.width, event.height),
                                           centering=(0.5, 0.5), color='white')
        self.canvas.img = PIL.ImageTk.PhotoImage(image=self.canvas.img)
        self.canvas.create_image(0, 0, image=self.canvas.img, anchor=tk.NW)


class ProcessingPages(tk.Frame):
    def update_images(self, image_paths):
        self.image_paths = image_paths

        for image_path in image_paths:
            page = ProcessingPage(self, self.controller, image_path)
            self.frames.append(page)

        self.show_frame(0)

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.image_paths: list = []
        self.frames: list = []

        self.grid_rowconfigure(0, weight=1, uniform="y")
        self.grid_columnconfigure(0, weight=1, uniform="x")

    def show_frame(self, i):
        frame = self.frames[i]
        frame.grid(row=0, column=0, sticky="nsew")
        self.controller.wm_title(self.image_paths[i])


@dataclass
class PathCanvasImageSelectedContainer:
    path: str
    canvas: tk.Canvas
    image: PIL.ImageTk.PhotoImage
    selected: bool


class SelectImagesPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.progressbar = ttk.Progressbar(self, orient='horizontal', mode='determinate')

        self.data: list[PathCanvasImageSelectedContainer] = []
        self.scroll_frame = ScrollFrame(self)
        self.images_frame = self.scroll_frame.viewPort

        def SelectImages():
            filetypes = (
                ('JPEG FILES', '*.jpg'),
                ('JPEG FILES', '*.jpeg'),
                ('PNG FILES', '*.png'),
                ('HEIF FILES', '*.heif'),
                ('HEIF FILES', '*.heic'),
                ('PDF FILES', '*.pdf'),
                ('All files', '*.*')
            )

            filepaths = fd.askopenfilenames(
                title='Select image files to process.',
                initialdir='~',
                filetypes=filetypes)

            self.update_images(filepaths)

        select_pictures_button = tk.Button(
            self,
            text='Select image files to process.',
            command=SelectImages
        )

        def SelectImagesDirectory():
            dirname = fd.askdirectory(title='Select directory whose image files are to be processed.',
                                      initialdir='~')

            if len(dirname) > 0:
                filepaths = [os.path.join(dirname, file) for file in os.listdir(dirname) if
                             file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png") or file.endswith(
                                 ".heif") or file.endswith(".heic")]
                self.update_images(filepaths)

        select_pictures_directory_button = tk.Button(
            self,
            text='Select directory (not recursive) whose image files are to be processed.',
            command=SelectImagesDirectory
        )

        def DeleteSelectedImages():
            for element in [element for element in self.data if element.selected]:
                element.canvas.destroy()
                self.data.remove(element)

        delete_selected_button = tk.Button(
            self,
            text='Delete selected images.',
            command=DeleteSelectedImages
        )

        def StartProcessing():
            if self.data:
                selected = [element.path for element in self.data if element.selected]
                if selected:
                    controller.show_frame(ProcessingPages).update_images(selected)
                else:
                    controller.show_frame(ProcessingPages).update_images([element.path for element in self.data])

        start_button = tk.Button(
            self,
            text='Start operation',
            command=StartProcessing
        )

        self.grid_rowconfigure(0)
        self.grid_rowconfigure(1)
        self.grid_rowconfigure(2, weight=1, uniform="x")
        self.grid_rowconfigure(2)
        self.grid_columnconfigure(0, weight=1, uniform="x")
        self.grid_columnconfigure(1, weight=1, uniform="x")

        select_pictures_button.grid(row=0, column=0, sticky="nswe")
        select_pictures_directory_button.grid(row=0, column=1, sticky="nsew")
        self.progressbar.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.scroll_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        delete_selected_button.grid(row=3, column=0, sticky="nsew")
        start_button.grid(row=3, column=1, sticky="nsew")

    def update_images(self, new_image_files):
        image_paths = list(set([element.path for element in self.data]) | set(new_image_files))
        self.progressbar['maximum'] = len(image_paths)
        self.progressbar['value'] = 0

        for element in self.data:
            element.canvas.destroy()

        self.data.clear()

        for i, image_path in enumerate(image_paths):
            self.progressbar['value'] += 1
            self.progressbar.update_idletasks()
            canvas = tk.Canvas(self.images_frame, width=248, height=248, background='white', highlightthickness=1,
                               highlightbackground="black")

            img = PIL.Image.open(image_path)
            img.thumbnail((canvas.winfo_reqwidth(), canvas.winfo_reqheight()), PIL.Image.Resampling.BILINEAR)
            img = PIL.ImageTk.PhotoImage(img)
            canvas.create_image(canvas.winfo_reqwidth() // 2, canvas.winfo_reqheight() // 2, image=img,
                                anchor=tk.CENTER)

            def select_image(event):
                for element in self.data:
                    if event.widget == element.canvas:
                        if element.selected:
                            element.selected = False
                            element.canvas.configure(highlightbackground="black")
                        else:
                            element.selected = True
                            element.canvas.configure(highlightbackground="red")
                        break

            canvas.bind("<Button-1>", select_image)

            canvas.grid(row=i // 5, column=i % 5, sticky="nsew", padx=1, pady=1)
            self.data.append(PathCanvasImageSelectedContainer(image_path, canvas, img, False))


class ScrollFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)  # create a frame (self)

        self.canvas = tk.Canvas(self, borderwidth=0)  # place canvas on self
        self.viewPort = tk.Frame(self.canvas)  # place a frame on the canvas, this frame will hold the child widgets
        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)  # place a scrollbar on self
        self.canvas.configure(yscrollcommand=self.vsb.set)  # attach scrollbar action to scroll of canvas

        self.vsb.pack(side="right", fill="y")  # pack scrollbar to right of self
        self.canvas.pack(side="left", fill="both", expand=True)  # pack canvas to left of self and expand to fil
        self.canvas_window = self.canvas.create_window((4, 4), window=self.viewPort, anchor="nw",
                                                       # add view port frame to canvas
                                                       tags="self.viewPort")

        self.viewPort.bind("<Configure>",
                           self.onFrameConfigure)  # bind an event whenever the size of the viewPort frame changes.
        self.canvas.bind("<Configure>",
                         self.onCanvasConfigure)  # bind an event whenever the size of the canvas frame changes.

        self.viewPort.bind('<Enter>', self.onEnter)  # bind wheel events when the cursor enters the control
        self.viewPort.bind('<Leave>', self.onLeave)  # unbind wheel events when the cursorl leaves the control

        self.onFrameConfigure(
            None)  # perform an initial stretch on render, otherwise the scroll region has a tiny border until the first resize

    def onFrameConfigure(self, event):
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox(
            "all"))  # whenever the size of the frame changes, alter the scroll region respectively.

    def onCanvasConfigure(self, event):
        '''Reset the canvas window to encompass inner frame when required'''
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window,
                               width=canvas_width)  # whenever the size of the canvas changes alter the window region respectively.

    def onMouseWheel(self, event):  # cross platform scroll wheel event
        if platform.system() == 'Windows':
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        elif platform.system() == 'Darwin':
            self.canvas.yview_scroll(int(-1 * event.delta), "units")
        else:
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")

    def onEnter(self, event):  # bind wheel events when the cursor enters the control
        if platform.system() == 'Linux':
            self.canvas.bind_all("<Button-4>", self.onMouseWheel)
            self.canvas.bind_all("<Button-5>", self.onMouseWheel)
        else:
            self.canvas.bind_all("<MouseWheel>", self.onMouseWheel)

    def onLeave(self, event):  # unbind wheel events when the cursorl leaves the control
        if platform.system() == 'Linux':
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
        else:
            self.canvas.unbind_all("<MouseWheel>")


if __name__ == "__main__":
    testObj = App()
    testObj.mainloop()
