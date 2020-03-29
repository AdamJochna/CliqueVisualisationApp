from tkinter import *
from PIL import Image, ImageTk
import random
import colorsys
import copy
import numpy as np
from ensembling_algos import algo_wrapper


class Paint(object):

    def __init__(self):

        self.size = 300
        self.root = Tk()

        self.add_button = Button(self.root, text='add', command=self.add_use)
        self.add_button.grid(row=0, column=0)

        self.move_button = Button(self.root, text='move', command=self.move_use)
        self.move_button.grid(row=0, column=1)

        self.c = Canvas(self.root, bg='white', width=4 * self.size, height=self.size)
        self.c.grid(row=1, columnspan=5)

        self.bg_img = Image.open('bg.png')
        self.bg_img = self.bg_img.resize((self.size, self.size), Image.ANTIALIAS)
        self.bg_img = ImageTk.PhotoImage(self.bg_img)
        self.tmpimages = []

        self.boxes = []
        self.pickedbox = None
        self.old_x = None
        self.old_y = None

        self.active_button = self.add_button
        self.active_button.config(relief=SUNKEN)

        self.c.bind('<Button-1>', self.click1)
        self.c.bind('<ButtonRelease-1>', self.click2)
        self.c.bind('<B1-Motion>', self.process_move)

        # for windows root.bind("<MouseWheel>", mouse_wheel)

        self.c.bind("<Button-4>", self.process_scroll)
        self.c.bind("<Button-5>", self.process_scroll)

        self.printcanvas()
        self.root.mainloop()

    def process_scroll(self, event):
        if len(self.boxes) > 0:
            if event.num == 5 or event.delta == -120:
                self.boxes[-1][5] = max(0.0, min(1.0, self.boxes[-1][5] - 0.02))
            if event.num == 4 or event.delta == 120:
                self.boxes[-1][5] = max(0.0, min(1.0, self.boxes[-1][5] + 0.02))

        self.printcanvas()

    def activate_button(self, some_button):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button

    def add_use(self):
        self.activate_button(self.add_button)

    def move_use(self):
        self.activate_button(self.move_button)

    def click1(self, event):
        self.old_x = event.x
        self.old_y = event.y

        if self.check_range(event.x, event.y):
            if self.active_button == self.move_button:
                index = self.get_corresponding_box(event.x, event.y)
                if index != -1:
                    self.pickedbox = self.boxes.pop(index)
                    self.boxes.append(self.pickedbox)
                    self.printcanvas()

    def click2(self, event):
        if self.check_range(event.x, event.y) and self.check_range(self.old_x, self.old_y):
            if self.active_button == self.add_button:
                if abs(self.old_x - event.x) > 10 and abs(self.old_y - event.y) > 10:
                    self.boxes.append([min(self.old_x, event.x), min(self.old_y, event.y), abs(self.old_x - event.x),
                                       abs(self.old_y - event.y), self.random_color(), np.random.random()])
                self.printcanvas()

            if self.active_button == self.move_button:
                self.pickedbox = None

    def process_move(self, event):
        if self.active_button == self.move_button and self.pickedbox != None:
            tmpbox = copy.deepcopy(self.pickedbox)
            tmpbox[0] = tmpbox[0] + (event.x - self.old_x)
            tmpbox[1] = tmpbox[1] + (event.y - self.old_y)

            tmpbox = self.corect_range(tmpbox)
            self.boxes[-1] = tmpbox

            self.printcanvas()

    def get_corresponding_box(self, x, y):
        index = -1

        for idx, box in enumerate(reversed(self.boxes)):
            if (box[0] <= x) and (box[0] + box[2] >= x) and (box[1] <= y) and (box[1] + box[3] >= y):
                index = len(self.boxes) - 1 - idx
                break

        return index

    def create_rectangle(self, x1, y1, x2, y2, **kwargs):
        if 'alpha' in kwargs:
            alpha = int(kwargs.pop('alpha') * 255)
            fill = kwargs.pop('fill')
            fill = fill + (alpha,)
            image = Image.new('RGBA', (x2 - x1, y2 - y1), fill)
            self.tmpimages.append(ImageTk.PhotoImage(image))
            self.c.create_image(x1, y1, image=self.tmpimages[-1], anchor='nw')
        self.c.create_rectangle(x1, y1, x2, y2, **kwargs)

    def printcanvas(self):
        self.c.delete("all")
        for i in range(4):
            self.c.create_image(i * self.size, 0, image=self.bg_img, anchor=NW)

        self.c.create_text(self.size//2 + 0 * self.size, 20, text="BOXES", fill='black')
        self.c.create_text(self.size//2 + 1 * self.size, 20, text="ENSEMBLE_CLIQUES", fill='black')
        self.c.create_text(self.size//2 + 2 * self.size, 20, text="NMS", fill='black')
        self.c.create_text(self.size//2 + 3 * self.size, 20, text="SOFT_NMS", fill='black')

        self.tmpimages = []

        for box in self.boxes:
            self.create_rectangle(box[0], box[1], box[0] + box[2], box[1] + box[3], fill=box[4], alpha=0.35)
            self.c.create_text(box[0] + 20, box[1] + 8, text="{0:.3f}".format(box[5]), fill='white')

        if len(self.boxes) > 0:
            results = algo_wrapper(copy.deepcopy(self.boxes))

            for idx, algoresult in enumerate(results):
                for box in algoresult:
                    if box[5] > 0.02:
                        self.create_rectangle(box[0] + (idx + 1) * self.size, box[1],
                                              box[0] + (idx + 1) * self.size + box[2], box[1] + box[3], fill=box[4], alpha=0.35)

                        self.c.create_text(box[0] + (idx + 1) * self.size + 20, box[1] + 8, text="{0:.3f}".format(box[5]), fill='white')

    def check_range(self, x, y):
        return (x >= 0 and x <= self.size and y >= 0 and y <= self.size)

    def corect_range(self, box):
        for i in range(2):
            if box[i] < 0:
                box[i] = 0
            if box[i] + box[i + 2] > self.size:
                box[i] = self.size - box[i + 2]

        return box

    def random_color(self):
        h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
        r, g, b = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
        return (r, g, b)


if __name__ == '__main__':
    Paint()
