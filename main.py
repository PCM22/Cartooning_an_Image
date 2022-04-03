import cv2
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

DIR = 'Img/img1.JPG'
win_size = '1300x800'


# show cv2 image
def show_image(image, percent=100):
    r = percent / 100
    width = int(image.shape[1] * r)
    height = int(image.shape[0] * r)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('{0} {1}x{2}'.format('image', width, height), resized)
    cv2.waitKey(0)


# get_resize_img
def get_resize_img(img, width=800, height=1000):
    if img.shape[0] > img.shape[1]:
        r = img.shape[0] / width
    else:
        r = img.shape[1] / height
    dim = (int(img.shape[1] / r), int(img.shape[0] / r))
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    return resized


# open function
def open_fn():
    # file type
    filetypes = (
        ('JPG file', '*.jpg'),
        ('PNG file', '*.png'),
        ('All files', '*.*')
    )
    return fd.askopenfile(filetypes=filetypes).name


# Save function
def save_fn():
    filetypes = [
        ('JPG file', '*.jpg'),
        ('PNG file', '*.png'),
        ('All files', '*.*')
    ]
    return fd.asksaveasfile(filetypes=filetypes, defaultextension=filetypes)


# EDIT PHOTO FUNCTION

def edge_mask(img, line_size=7, blur_value=7):
    """Create Edge Mask"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size,
                                  blur_value)
    return edges


def color_quantization(img, total_color=9):
    # Transform the image
    data = np.float32(img).reshape((-1, 3))
    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implementing K-Means
    ret, label, center = cv2.kmeans(data, total_color, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result


def color_quantization_blur(img, d=7, sigmaColor=200, sigmaSpace=200, total_color=9):
    quant = color_quantization(img, total_color)
    blurred = cv2.bilateralFilter(quant, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return blurred


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # variable
        self.image = None
        self.img_change = None
        self.img_mask = None
        self.quant_mask = None
        self.quant = None
        self.path = None
        self.panelA = None

        self.var1 = tk.IntVar()
        self.var2 = tk.IntVar()
        self.var3 = tk.IntVar()
        self.line_size = tk.IntVar(value=7)
        self.blur_value = tk.IntVar(value=7)
        self.total_color = tk.IntVar(value=9)
        self.d = tk.IntVar(value=7)
        self.sigmaColor = tk.IntVar(value=200)
        self.sigmaSpace = tk.IntVar(value=200)

        # configure the root window
        global win_size
        self.title('Image to carton')
        self.resizable(False, False)
        self.geometry(win_size)

        self.frame1 = tk.Frame(self, width=1000, height=800)
        self.frame1.pack(side='left')

        self.frame2 = tk.Frame(self)
        self.frame2.pack(side='right')

        #################################################################################################

        # BUTTON

        open_button = ttk.Button(self.frame2, text='Open a File')
        open_button['command'] = lambda: self.open_img()
        open_button.grid(row=9, column=1)

        apply_button = ttk.Button(self.frame2, text='Apply')
        apply_button['command'] = lambda: self.apply()
        apply_button.grid(row=9, column=0)

        save_button = ttk.Button(self.frame2, text='Save As')
        save_button['command'] = lambda: self.save_as()
        save_button.grid(row=10, column=0, padx=(0, 20), pady=(20, 0), columnspan=2)

        #################################################################################################
        # LABEL WIDGETS

        # Create Edge Mask
        l1 = tk.Label(self.frame2, text="Line size")
        l2 = tk.Label(self.frame2, text="Blur value")
        l1.grid(row=1, column=0, sticky='w', pady=2, padx=(0, 10))
        l2.grid(row=2, column=0, sticky='w', pady=2, padx=(0, 10))

        # Reduce the Color Palette
        l4 = tk.Label(self.frame2, text="Number_colors")
        l4.grid(row=4, column=0, sticky='w', pady=2, padx=(0, 10))

        # Bilateral Filter
        l6 = tk.Label(self.frame2, text="Diameter")
        l7 = tk.Label(self.frame2, text="SigmaColor")
        l8 = tk.Label(self.frame2, text="SigmaSpace ")
        l6.grid(row=6, column=0, sticky='w', pady=2, padx=(0, 10))
        l7.grid(row=7, column=0, sticky='w', pady=2, padx=(0, 10))
        l8.grid(row=8, column=0, sticky='w', pady=2, padx=(0, 10))

        #################################################################################################
        # ENTRY WIDGETS

        # Edge Mask
        e1 = tk.Entry(self.frame2, textvariable=self.line_size)
        e2 = tk.Entry(self.frame2, textvariable=self.blur_value)
        e1.grid(row=1, column=1, pady=2, padx=(0, 30))
        e2.grid(row=2, column=1, pady=2, padx=(0, 30))

        # Reduce the Color Palette
        e3 = tk.Entry(self.frame2, textvariable=self.total_color)
        e3.grid(row=4, column=1, pady=2, padx=(0, 30))

        # Bilateral Filter
        e4 = tk.Entry(self.frame2, textvariable=self.d)
        e5 = tk.Entry(self.frame2, textvariable=self.sigmaColor)
        e6 = tk.Entry(self.frame2, textvariable=self.sigmaSpace)
        e4.grid(row=6, column=1, pady=2, padx=(0, 30))
        e5.grid(row=7, column=1, pady=2, padx=(0, 30))
        e6.grid(row=8, column=1, pady=2, padx=(0, 30))
        ###################################################################################################
        # Checkbutton

        c1 = tk.Checkbutton(self.frame2, font="Arial 12 bold", text='Create Edge Mask', variable=self.var1,
                            onvalue=1, offvalue=0, command=lambda: print(1))
        c1.grid(row=0, column=0, sticky='w', pady=(10, 2), columnspan=2)

        c2 = tk.Checkbutton(self.frame2, font="Arial 12 bold", text='Color_quantization', variable=self.var2,
                            onvalue=1, offvalue=0, command=lambda: print(2))
        c2.grid(row=3, column=0, sticky='w', pady=(10, 2), columnspan=2)

        c3 = tk.Checkbutton(self.frame2, font="Arial 12 bold", text='Bilateral Filter', variable=self.var3,
                            onvalue=1, offvalue=0, command=lambda: print(3))
        c3.grid(row=5, column=0, sticky='w', pady=(10, 2), columnspan=2)

    ###################################################################################################
    # GUI FUNCTION
    def apply(self):
        if self.var1.get() & self.var3.get():
            blurred = color_quantization_blur(self.image,
                                              d=self.d.get(),
                                              sigmaColor=self.sigmaColor.get(),
                                              sigmaSpace=self.sigmaSpace.get(),
                                              total_color=self.total_color.get())
            edges = edge_mask(self.image, line_size=self.line_size.get(), blur_value=self.blur_value.get())
            self.img_change = cv2.bitwise_and(blurred, blurred, mask=edges)

        elif self.var3.get():
            self.img_change = color_quantization_blur(self.image,
                                                      d=self.d.get(),
                                                      sigmaColor=self.sigmaColor.get(),
                                                      sigmaSpace=self.sigmaSpace.get(),
                                                      total_color=self.total_color.get())

        elif self.var1.get() & self.var2.get():
            edges = edge_mask(self.image, line_size=self.line_size.get(), blur_value=self.blur_value.get())
            blurred = color_quantization(self.image, total_color=self.total_color.get())
            self.img_change = cv2.bitwise_and(blurred, blurred, mask=edges)

        elif self.var2.get():
            self.quant = color_quantization(self.image, total_color=self.total_color.get())
            self.img_change = self.quant

        elif self.var1.get():
            self.img_mask = edge_mask(self.image, line_size=self.line_size.get(), blur_value=self.blur_value.get())
            self.img_change = self.img_mask

        self.img_update(image=self.img_change)

    def img_update(self, image):
        """Load the image into the interface"""
        image = get_resize_img(image, width=800, height=1000)
        # convert the images to PIL format and then to ImageTk format
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        # Load the image
        if self.panelA is None:
            self.frame1.config(bg='white')
            self.panelA = ttk.Label(self.frame1, image=image)
            self.panelA.image = image
            self.panelA.pack(side="left", padx=10, pady=10)
        else:
            self.panelA.configure(image=image)
            self.panelA.image = image

    def open_img(self):
        self.path = open_fn()
        if len(self.path) > 0:
            self.image = cv2.imread(self.path)
            # We keep the original and work with the copy
            self.img_change = self.image
            # convert color from BGR format to RGB format
            self.img_change = cv2.cvtColor(self.img_change, cv2.COLOR_BGR2RGB)
            self.img_update(self.img_change)

    def save_as(self):
        img = Image.fromarray(self.img_change)
        destination = save_fn().name
        img.save(destination, "JPEG")

    def run(self):
        self.mainloop()


if __name__ == '__main__':
    Window = App()
    Window.run()
