# AGH UST Medical Informatics 03.2021
# Lab 2 : DICOM

import pydicom
from tkinter import *
import numpy as np
from PIL import Image, ImageTk


class MainWindow():

    ds = pydicom.dcmread("head.dcm")
    data = ds.pixel_array

    def __init__(self, main):
        # print patient name
        print(self.ds.PatientName)

        # todo: from ds get windowWidth and windowCenter
        self.win_center = self.ds.WindowCenter
        self.win_width = self.ds.WindowWidth
        print(f"Window: width={self.win_width}, center={self.win_center}")

        # distance measurement
        self.line = None
        self.prev_mouse = None
        self.spacing = self.ds.PixelSpacing  # in mm/px

        # prepare canvas
        self.canvas = Canvas(main, width=512, height=512)
        self.canvas.grid(row=0, column=0)
        self.canvas.bind("<Button-1>", self.init_window)
        self.canvas.bind("<B1-Motion>", self.update_window)
        self.canvas.bind("<Button-3>", self.init_measurement)
        self.canvas.bind("<B3-Motion>", self.update_measurement)
        self.canvas.bind("<ButtonRelease-3>", self.finish_measurement)

        # load image
        # todo: apply transform
        self.array = self.transform_data(
            self.data, self.win_width, self.win_center)
        self.image = Image.fromarray(self.array)
        self.image = self.image.resize((512, 512), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(image=self.image, master=root)
        self.image_on_canvas = self.canvas.create_image(
            0, 0, anchor=NW, image=self.img)

    def transform_data(self, data, window_width, window_center):
        img_min = max(0, window_center - window_width//2)
        img_max = window_center + window_width//2
        cp = data.copy()
        cp[data < img_min] = img_min
        cp[data > img_max] = img_max
        cp = (cp - img_min) / (img_max - img_min) * 255
        return cp

    def init_window(self, event):
        pass

    def update_window(self, event):
        width = self.win_width * event.x // self.data.shape[0]
        center = self.win_center * event.y // self.data.shape[1]
        self.array2 = self.transform_data(self.data, width, center)
        self.image2 = Image.fromarray(self.array2)
        self.image2 = self.image2.resize((512, 512), Image.ANTIALIAS)
        self.img2 = ImageTk.PhotoImage(image=self.image2, master=root)
        self.canvas.itemconfig(self.image_on_canvas, image=self.img2)

    def init_measurement(self, event):
        self.prev_mouse = event.x, event.y
        self.line = self.canvas.create_line(
            event.x, event.y, event.x, event.y, fill="red", width=3)

    def update_measurement(self, event):
        self.canvas.coords(
            self.line, self.prev_mouse[0], self.prev_mouse[1], event.x, event.y)

    def finish_measurement(self, event):
        diff_x = event.x - self.prev_mouse[0]
        diff_y = event.y - self.prev_mouse[1]
        dist = np.sqrt(
            (diff_x * self.spacing[0])**2 + (diff_y * self.spacing[1])**2)
        print(f"Measured {round(dist, 3)}mm")
        angle = np.arctan2(diff_x, diff_y) * 180 / np.pi + 90
        if angle > 90 or angle > 270:
            angle = angle + 180

        mid_x = event.x - diff_x / 2 - 13 * np.sin(angle / 180 * np.pi)
        mid_y = event.y - diff_y / 2 - 13 * np.cos(angle / 180 * np.pi)
        self.canvas.create_text(mid_x, mid_y, anchor="center",
                                text=f"{round(dist)}mm", angle=angle, fill="red", font='Helvetica 10 bold', justify="center")


# ----------------------------------------------------------------------
root = Tk()
MainWindow(root)
root.mainloop()
