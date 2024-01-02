import tkinter as tk
from tkinter import filedialog as fd
from typing import Tuple, Dict
from pathlib import Path
import PIL.Image
import PIL.ImageTk
import PIL.ImageOps
import PIL.ImageDraw
import numpy as np
import os
import platform
import cv2
from pillow_heif import register_heif_opener
from dataclasses import dataclass
from tkinter import ttk
import multiprocessing
import math
from solutions import use_hog, use_mediapipe, use_cnn, face_align_crop
import multiprocessing.connection
import multiprocessing.queues

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
        self.grid_rowconfigure(0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.progressbar = ttk.Progressbar(self, orient='horizontal', mode='determinate')
        self.progressbar.grid(row=0, column=0, sticky="nsew")

        # Initialize container frame which can be switched
        self.container = tk.Frame(self)
        self.container.grid(row=1, column=0, sticky="nsew")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for Frame in (SelectImagesPage, ProcessingPages, EndPage):
            self.frames[Frame] = Frame(self.container, self)
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
    image: PIL.Image
    selected: bool


class EndPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.label: tk.Label = tk.Label(self, text="End")
        self.label.pack(fill="both")

        def restart():
            self.controller.show_frame(SelectImagesPage)

        self.button: tk.Button = tk.Button(self, text="Restart", command=restart)
        self.button.pack(fill="both")


class ProcessingPage(tk.Frame):
    def show(self):
        detection_data = self.parent_conn.recv()
        self.process.join()
        for detection_image, x_start_bbox, y_start_bbox, x_end_bbox, y_end_bbox in detection_data:
            canvas = tk.Canvas(self.images_frame, width=self.controller.output_width - 2,
                               height=self.controller.output_height - 2, background='white', highlightthickness=1,
                               highlightbackground="black")

            canvas.img = PIL.ImageTk.PhotoImage(detection_image)
            canvas.create_image(0, 0, image=canvas.img, anchor=tk.NW)

            def select_image(selection_event):
                for element in self.data:
                    if selection_event.widget == element.canvas:
                        if element.selected:
                            element.selected = False
                            element.canvas.configure(highlightbackground="black")
                        else:
                            element.selected = True
                            element.canvas.configure(highlightbackground="green")
                    else:
                        element.selected = False
                        element.canvas.configure(highlightbackground="black")

            canvas.bind("<Button-1>", select_image)

            canvas.grid(row=len(self.data), sticky="nsew", padx=1, pady=1)
            if not len(self.data):
                canvas.configure(highlightbackground="green")
            self.data.append(CanvasImageSelectedContainer(canvas, detection_image, False if len(self.data) else True))
            draw_img = PIL.ImageDraw.Draw(self.draw_image)
            draw_img.rectangle(((x_start_bbox, y_start_bbox), (x_end_bbox, y_end_bbox)), outline='red',
                               width=(self.draw_image.width + self.draw_image.height) // 1000 + 1)

        @dataclass
        class Event:
            width: int
            height: int

        event = Event(width=self.canvas.winfo_width(), height=self.canvas.winfo_height())
        self.onCanvasConfigure(event)

    @staticmethod
    def process_image(func, image: PIL.Image, output_width: int, output_height: int,
                      connection: multiprocessing.connection.Connection):
        print("Processing image...")
        results = [(face_align_crop(image, output_width, output_height, *args), args[0], args[1], args[2], args[3])
                   for args in func(image)]
        print(results)
        connection.send(results)
        connection.close()

    def __init__(self, parent, controller, image_path):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.image_path = image_path
        self.image = PIL.Image.open(self.image_path)
        self.draw_image = self.image.copy()

        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=self.process_image, args=(
            use_mediapipe, self.image, self.controller.output_width, self.controller.output_height, self.child_conn),
                                               daemon=True)
        self.process.start()

        self.data: list[CanvasImageSelectedContainer] = []

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1)
        self.grid_columnconfigure(0, weight=1, uniform="x")
        self.grid_columnconfigure(1, minsize=controller.output_width + 20)

        self.canvas = tk.Canvas(self, highlightthickness=1, highlightbackground="black")
        self.canvas.grid(row=0, column=0, padx=2, pady=2, sticky="nswe")
        self.canvas.bind("<Configure>", self.onCanvasConfigure)

        self.scroll_frame = ScrollFrame(self)
        self.images_frame = self.scroll_frame.viewPort
        self.scroll_frame.grid(row=0, column=1, sticky="nswe")

        self.frame_buttons = tk.Frame(self)
        self.frame_buttons.grid_rowconfigure(0)
        self.frame_buttons.grid_columnconfigure(0, weight=1, uniform="x")
        self.frame_buttons.grid_columnconfigure(1, weight=1, uniform="x")
        self.frame_buttons.grid_columnconfigure(2, weight=1, uniform="x")
        self.frame_buttons.grid_columnconfigure(3, weight=1, uniform="x")

        print(self.image_path)

        self.button_mediapipe = tk.Button(self.frame_buttons, text="Use mediapipe")
        self.button_mediapipe.grid(row=0, column=0, sticky="nswe")
        self.button_mediapipe["state"] = "disabled"

        def hog_click():
            self.button_hog["state"] = "disabled"
            self.button_hog.update()

            self.parent_conn, self.child_conn = multiprocessing.Pipe()
            self.process = multiprocessing.Process(target=self.process_image, args=(
                use_hog, self.image, self.controller.output_width, self.controller.output_height,
                self.child_conn), daemon=True)
            self.process.start()

            self.parent.next_frame()

        self.button_hog = tk.Button(self.frame_buttons, text="Use dlib hog", command=hog_click)
        self.button_hog.grid(row=0, column=1, sticky="nswe")

        def cnn_click():
            self.button_cnn["state"] = "disabled"
            self.button_cnn.update()

            self.parent_conn, self.child_conn = multiprocessing.Pipe()
            self.process = multiprocessing.Process(target=self.process_image, args=(
                use_cnn, self.image, self.controller.output_width, self.controller.output_height,
                self.child_conn), daemon=True)
            self.process.start()

            self.parent.next_frame()

        self.button_cnn = tk.Button(self.frame_buttons, text="Use dlib cnn", command=cnn_click)
        self.button_cnn.grid(row=0, column=2, sticky="nswe")

        def save_click():
            self.button_save["state"] = "disabled"
            self.button_save.update()

            for container in self.data:
                if container.selected:
                    desktop = os.path.normpath(os.path.expanduser("~/Desktop"))
                    os.makedirs(os.path.join(desktop, "ready"), exist_ok=True)
                    container.image.save(os.path.join(desktop, "ready", Path(self.image_path).stem + str(".png")))
                    break

            self.parent.destroy_frame()

        self.button_save = tk.Button(self.frame_buttons, text="save selection", command=save_click)
        self.button_save.grid(row=0, column=3, sticky="nswe")

        self.frame_buttons.grid(row=1, column=0, columnspan=2, sticky="nswe")

    def onCanvasConfigure(self, event):
        self.canvas.img = PIL.ImageOps.pad(self.draw_image, (event.width, event.height),
                                           centering=(0.5, 0.5), color='white')
        self.canvas.img = PIL.ImageTk.PhotoImage(image=self.canvas.img)
        self.canvas.create_image(0, 0, image=self.canvas.img, anchor=tk.NW)


@dataclass
class FramePathContainer:
    frame: ProcessingPage
    image_path: str


class ProcessingPages(tk.Frame):
    def update_images(self, image_paths):
        self.controller.progressbar['maximum'] = len(image_paths)
        self.controller.progressbar['value'] = 0

        for image_path in image_paths:
            self.controller.progressbar['value'] += 1
            self.controller.progressbar.update()
            page = ProcessingPage(self, self.controller, image_path)
            self.frames.append(FramePathContainer(page, image_path))

        self.show_frame()

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.frames: list[FramePathContainer] = []
        self.current_frame = 0

        self.grid_rowconfigure(0, weight=1, uniform="y")
        self.grid_columnconfigure(0, weight=1, uniform="x")

    def show_frame(self):
        frame = self.frames[self.current_frame]
        frame.frame.grid(row=0, column=0, sticky="nsew")
        self.controller.wm_title(frame.image_path)
        frame.frame.show()
        frame.frame.tkraise()
        self.update()

    def destroy_frame(self):
        self.frames[self.current_frame].frame.destroy()
        self.frames.pop(self.current_frame)
        if len(self.frames):
            self.current_frame = self.current_frame % len(self.frames)
            self.show_frame()
        else:
            self.controller.show_frame(EndPage)

    def next_frame(self):
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        self.show_frame()


@dataclass
class PathCanvasImageSelectedContainer:
    path: str
    canvas: tk.Canvas
    image: PIL.ImageTk.PhotoImage
    selected: bool


class SelectImagesPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

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
                # Only use filepaths that have an supported image file extension
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

                # Use all images or only the selected ones
                if selected:
                    controller.show_frame(ProcessingPages).update_images(selected)
                else:
                    controller.show_frame(ProcessingPages).update_images([element.path for element in self.data])

                # Delete images on page to allow restarting
                for element in self.data:
                    element.canvas.destroy()
                self.data.clear()

        start_button = tk.Button(
            self,
            text='Start operation',
            command=StartProcessing
        )

        self.grid_rowconfigure(0)
        self.grid_rowconfigure(1, weight=1, uniform="x")
        self.grid_rowconfigure(2)
        self.grid_columnconfigure(0, weight=1, uniform="x")
        self.grid_columnconfigure(1, weight=1, uniform="x")

        select_pictures_button.grid(row=0, column=0, sticky="nswe")
        select_pictures_directory_button.grid(row=0, column=1, sticky="nsew")
        self.scroll_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        delete_selected_button.grid(row=2, column=0, sticky="nsew")
        start_button.grid(row=2, column=1, sticky="nsew")

    def update_images(self, new_image_files):
        '''
        Displays images from the provided file list in a grid.
        '''
        image_paths = list(set([element.path for element in self.data]) | set(new_image_files))
        self.controller.progressbar['maximum'] = len(image_paths)
        self.controller.progressbar['value'] = 0

        for element in self.data:
            element.canvas.destroy()

        self.data.clear()

        for i, image_path in enumerate(image_paths):
            self.controller.progressbar['value'] += 1
            self.controller.progressbar.update()
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