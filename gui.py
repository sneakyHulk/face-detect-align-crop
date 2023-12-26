import tkinter as tk
from tkinter import filedialog as fd
import PIL.Image
import PIL.ImageTk
import PIL.ImageOps
import numpy as np
import os
import platform
import cv2
from PIL import ImageTk
# from pi_heif import register_heif_opener
from dataclasses import dataclass
from tkinter import ttk


# register_heif_opener()


def pathtoTkinter(image_path):
    cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return cv2toTkinter(cv_img)


def cv2toTkinter(cv_img: np.ndarray):
    return PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))


class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

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


class ProcessingPage(tk.Frame):
    def __init__(self, parent, image_path):
        tk.Frame.__init__(self, parent)
        self.image_path = image_path

        cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        height, width, _ = cv_img.shape

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        canvas = tk.Canvas(self, background="white")
        canvas.grid(row=0, column=0, sticky="nsew")

        self.img = PIL.Image.fromarray(cv_img)
        if width < canvas.winfo_reqwidth() or height < canvas.winfo_reqheight():
            self.img = PIL.ImageOps.scale(self.img,
                                          max(width / canvas.winfo_reqwidth(), height / canvas.winfo_reqheight()))
        self.img = PIL.ImageOps.pad(self.img, (canvas.winfo_reqheight(), canvas.winfo_reqwidth()),
                                    centering=(0.5, 0.5))
        self.img = PIL.ImageTk.PhotoImage(image=self.img)
        canvas.create_image(0, 0, image=self.img,
                            anchor=tk.NW)

        # frame = tk.Frame(self)
        # frame.grid(row=0, column=0, sticky="NESW")

        ## Create a canvas that can fit the above image
        # canvas = tk.Canvas(frame, scrollregion=(0, 0, width, height), background="white")
        # horizontal_bar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        # horizontal_bar.pack(side=tk.BOTTOM, fill=tk.X)
        # horizontal_bar.config(command=canvas.xview)
        # vertical_bar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        # vertical_bar.pack(side=tk.RIGHT, fill=tk.Y)
        # vertical_bar.config(command=canvas.yview)
        # canvas.config(xscrollcommand=horizontal_bar.set, yscrollcommand=vertical_bar.set)
        # canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # self.img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))
        # canvas.create_image(0, 0, image=self.img, anchor=tk.NW)


class ProcessingPages(tk.Frame):
    def update_images(self, image_paths):
        self.image_paths = image_paths

        # for image_path in image_paths:
        #    page = ProcessingPage(self, image_path)
        #    self.frames.append(page)

        self.show_frame(0)

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.image_paths: list = []
        self.frames: list = []

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(self, background="white")
        canvas.grid(row=0, column=0, sticky="nsew")

    def show_frame(self, i):
        cv_img = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)

        canvas = tk.Canvas(self, background="white")
        canvas.grid(row=0, column=0, sticky="nsew")

        self.img = PIL.Image.fromarray(cv_img)
        # self.img = PIL.ImageOps.pad(self.img, (1280 // 2, 734),
        self.img = PIL.ImageOps.pad(self.img, (canvas.winfo_reqwidth(), canvas.winfo_reqheight()),
                                    centering=(0.5, 0.5))

        self.img = PIL.ImageTk.PhotoImage(image=self.img)
        canvas.create_image(0, 0, image=self.img,
                            anchor=tk.NW)

        # frame = self.frames[i]
        # frame.grid(row=0, column=0, sticky="nsew")
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
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(2)
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)

        select_pictures_button.grid(row=0, column=0, sticky="nsew")
        select_pictures_directory_button.grid(row=0, column=1, sticky="nsew")
        self.progressbar.grid(row=1, column=0, columnspan=2, sticky="nsew")
        self.scroll_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        delete_selected_button.grid(row=3, column=0, sticky="nsew")
        start_button.grid(row=3, column=1, sticky="nsew")

    def update_images(self, new_image_files):
        image_paths = list(set([element.path for element in self.data]) | set(new_image_files))

        for element in self.data:
            element.canvas.destroy()

        self.data.clear()

        for i, image_path in enumerate(image_paths):
            canvas = tk.Canvas(self.images_frame, width=248, height=248, background='white', highlightthickness=1,
                               highlightbackground="black")

            img = PIL.Image.open(image_path)
            img.thumbnail((248, 248), PIL.Image.Resampling.BILINEAR)
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

            canvas.bind("<Button-1>", select_image)

            canvas.grid(row=i // 5, column=i % 5, sticky="nsew", padx=0.5, pady=0.5)
            self.data.append(PathCanvasImageSelectedContainer(image_path, canvas, img, False))


class ScrollFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)  # create a frame (self)

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
