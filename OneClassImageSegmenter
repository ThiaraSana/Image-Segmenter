import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path 

import numpy as np

from skimage import io

import os
from os import path

import glob

#Global Variables

VALID_IMAGE_TYPES = ['jpeg', 'png', 'bmp', 'gif', 'jpg']
FIGSIZE=(9,7)
OVERLAY_ALPHA=.5
OVERLAY_ERASE = 0
HEIGHT_OF_GUI = 30
WIDTH_OF_GUI = 1500
IMG_IDX = 0
INDICES = None
LINEPROPS = {'color': 'black', 'linewidth': 1, 'alpha': 0.8}
ZOOM_SCALE=1.1
NUMBEROFIMAGES = 0
IMAGE_PATHS = []
MASK_DIR = []
MASK_PATH = []
PIX = []
# CLASS_MASK = []
IMG = []

#Functions

def zoom_factory(ax,base_scale = 1.1):
    """
    parameters
    ----------
    ax : matplotlib axes object
        axis on which to implement scroll to zoom
    base_scale : float
        how much zoom on each tick of scroll wheel
 
    returns
    -------
    disconnect_zoom : function
        call this to disconnect the scroll listener
    """
    def limits_to_range(lim):
        return lim[1] - lim[0]
    
    fig = ax.get_figure() # get the figure of interest
    toolbar = fig.canvas.toolbar
    toolbar.push_current()
    orig_xlim = ax.get_xlim()
    orig_ylim = ax.get_ylim()
    orig_yrange = limits_to_range(orig_ylim)
    orig_xrange = limits_to_range(orig_xlim)
    orig_center = ((orig_xlim[0]+orig_xlim[1])/2, (orig_ylim[0]+orig_ylim[1])/2)

    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        # set the range
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            scale_factor = base_scale
        elif event.button == 'down':
            scale_factor = 1/base_scale
        else:
            scale_factor = 1

        new_xlim = [xdata - (xdata-cur_xlim[0]) / scale_factor,
                     xdata + (cur_xlim[1]-xdata) / scale_factor]
        new_ylim = [ydata - (ydata-cur_ylim[0]) / scale_factor,
                         ydata + (cur_ylim[1]-ydata) / scale_factor]
        new_yrange = limits_to_range(new_ylim)
        new_xrange = limits_to_range(new_xlim)

        if np.abs(new_yrange)>np.abs(orig_yrange):
            new_ylim = orig_center[1] -new_yrange/2 , orig_center[1] +new_yrange/2
        if np.abs(new_xrange)>np.abs(orig_xrange):
            new_xlim = orig_center[0] -new_xrange/2 , orig_center[0] +new_xrange/2
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)

    # attach the call back
    cid = fig.canvas.mpl_connect('scroll_event',zoom_fun)
    def disconnect_zoom():
        fig.canvas.mpl_disconnect(cid)    

    #return the disconnect function
    return disconnect_zoom

def GetFolder():
    global MASK_DIR, IMAGE_PATHS, INDICES, NUMBEROFIMAGES
    img_dir = filedialog.askdirectory()
    if not path.isdir(path.join(img_dir, 'train')):
        raise ValueError(f"{img_dir} must exist and contain the the folder 'train'")
    img_dir = path.join(img_dir, 'train')
    MASK_DIR = path.join(img_dir.rsplit('train_imgs/',1)[0], 'train_masks/train')
    if not os.path.isdir(MASK_DIR):
        os.makedirs(MASK_DIR)
    for type_ in VALID_IMAGE_TYPES:
        IMAGE_PATHS += (glob.glob(img_dir.rstrip('/')+f'/*.{type_}'))
        NUMBEROFIMAGES = len(IMAGE_PATHS)
    new_image(IMG_IDX)

def new_image(IMG_IDX):
    global  DISPLAYED, LASSO, IMG, CLASS_MASK,  PIX,  MASK_PATH
    plt.ion()
    shape = None
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.gca()
    IMG = io.imread(IMAGE_PATHS[IMG_IDX])
    img_path = IMAGE_PATHS[IMG_IDX]
    ax.set_title(os.path.basename(img_path))
    MASK_PATH = MASK_DIR + f'/{os.path.basename(img_path)}'
    shape = IMG.shape
    pix_x = np.arange(shape[0])
    pix_y = np.arange(shape[1])
    xv, yv = np.meshgrid(pix_y,pix_x)
    PIX = np.vstack( (xv.flatten(), yv.flatten()) ).T
    DISPLAYED = ax.imshow(IMG)
    
    Display_ImgIndex['text'] = 'Image Number: ' + str(root.counter) + '/' + str(NUMBEROFIMAGES)

    if os.path.exists(MASK_PATH):
        CLASS_MASK = io.imread(MASK_PATH)
    else:
        CLASS_MASK = np.zeros([shape[0],shape[1]],dtype=np.uint8)
    
    LASSO = LassoSelector(ax, InitializeLasso, lineprops=LINEPROPS)
    LASSO.set_active(True)
    
    if ZOOM_SCALE is not None:
            disconnect_scroll = zoom_factory(ax, base_scale = ZOOM_SCALE)

def save_mask(save_if_no_nonzero=False):
    if (save_if_no_nonzero or np.any(CLASS_MASK != 0)):
        if os.path.splitext(MASK_PATH)[1] in ['jpg', 'jpeg']:
            io.imsave(MASK_PATH, CLASS_MASK*255, check_contrast =False, quality=100)
        else:
            io.imsave(MASK_PATH, CLASS_MASK*255, check_contrast =False)

def updateArray(INDICES, ResetValue, WhichClass):

    array = DISPLAYED.get_array().data
    class_dropdown = WhichClass

    ImageClasses = [['Non-Perfusion Area'], ['Blockage Artefact'], ['High Standard Deviation Artefact'], ['Perfusion Area']]
    if isinstance(ImageClasses, int):
        ImageClasses = np.arange(ImageClasses)
    if len(ImageClasses)<=10:
        colors = 'tab10'
    elif len(ImageClasses)<=20:
        colors = 'tab20'
    else:
        raise ValueError(f'Currently only up to 20 ImageClasses are supported, you tried to use {len(ImageClasses)} ImageClasses')
    colors = np.vstack([[0,0,0],plt.get_cmap(colors)(np.arange(len(ImageClasses)))[:,:3]])
    
    if ResetValue ==1: ## To Reset
        CLASS_MASK[INDICES] = 0
        array[INDICES] = IMG[INDICES]
    elif class_dropdown == 0: ## To Erase
        CLASS_MASK[INDICES] = 0
        c_overlay = colors[CLASS_MASK[INDICES]]*255*OVERLAY_ERASE
        array[INDICES] = (c_overlay + IMG[INDICES]*(1-OVERLAY_ERASE))
    elif INDICES is not None: ## To Draw
        CLASS_MASK[INDICES] = class_dropdown
        c_overlay = colors[CLASS_MASK[INDICES]]*255*OVERLAY_ALPHA 
        array[INDICES] = (c_overlay + IMG[INDICES]*(1-OVERLAY_ALPHA))
    else:
        idx = CLASS_MASK != 0
        c_overlay = colors[CLASS_MASK[idx]]*255*OVERLAY_ALPHA
        array[idx] = (c_overlay + IMG[idx]*(1-OVERLAY_ALPHA))
    
    DISPLAYED.set_data(array)

def ResetFunction():
    if (ResetValue.get() ==1):
        updateArray(INDICES, ResetValue = 1, WhichClass=0)

def ClassFunction(*args):
    for ActionIndex,ActionType in ActionDictionary.items():
        if ActionType==options_draw.get():
            if ActionIndex == 1:
                for ClassIndex,ClassType in ClassDictionary.items():
                    if ClassType==options_class.get():
                        return ClassIndex           
            elif ActionIndex == 2: 
                ClassIndex = 0
                return ClassIndex

def InitializeLasso(verts):
    LassoPath = Path(verts)
    ClassIndex = ClassFunction()
    INDICES = LassoPath.contains_points(PIX, radius=0).reshape(450,540) #(1024,1024)
    updateArray(INDICES, ResetValue = 0, WhichClass = ClassIndex)

def _change_image_idx_Next():
    global IMG_IDX
    if IMG_IDX +1 < len(IMAGE_PATHS):
        IMG_IDX += 1
        root.counter += 1
        Display_ImgIndex['text'] = 'Image Number: ' + str(root.counter) + '/' + str(NUMBEROFIMAGES)
        save_mask()
        new_image(IMG_IDX)
        if IMG_IDX == len(IMAGE_PATHS):
            Next(disabled = True)
            messagebox.showerror("Error", "There are no more images in folder")


def _change_image_idx_Previous():
    global IMG_IDX
    if IMG_IDX>=1:
        IMG_IDX -= 1
        root.counter -= 1
        Display_ImgIndex['text'] = 'Image Number: ' + str(root.counter) + '/' + str(NUMBEROFIMAGES)
        save_mask()
        new_image(IMG_IDX)
        if IMG_IDX == 0:
            messagebox.showerror("Error", "You have reached the beginning of the folder.")
    
#Tkinter Window

if __name__=="__main__":
    root = tk.Tk()
    root.title("Image Segmenter")

    root.counter = 1

    ResetValue = tk.IntVar()
    EraseValue = tk.IntVar()
    ClassDictionary={1: 'Non-Perfusion Area', 2: 'Blockage Artefact', 3: 'High Standard Deviation Artefact', 4: 'Perfusion Area'}
    ActionDictionary={1: 'Draw', 2: 'Erase'}

    canvas = tk.Canvas(root, height=HEIGHT_OF_GUI, width=WIDTH_OF_GUI)
    canvas.pack(side=tk.RIGHT, fill = tk.BOTH, expand=2)

    ButtonFrame = tk.Frame(root)
    ButtonFrame.place(relx=0, rely=0, relwidth=1, relheight=1)

    options_class = tk.StringVar(ButtonFrame)
    options_class.set(ClassDictionary[1])
    options_draw = tk.StringVar(ButtonFrame)
    options_draw.set(ActionDictionary[1])

    LoadFolder = tk.Button(ButtonFrame, text="Load Folder", command = GetFolder)
    LoadFolder.place(relx=0.05, rely=0.05, relwidth=0.1, relheight=0.9)
    
    ClassDropDown = tk.OptionMenu( ButtonFrame, options_class, *ClassDictionary.values())
    ClassDropDown.place(relx=0.15, rely=0.05, relwidth=0.2, relheight=0.95)

    Action = tk.OptionMenu(ButtonFrame, options_draw, *ActionDictionary.values())
    Action.place(relx=0.35, rely=0.05, relwidth=0.1, relheight=0.9)
    
    Reset = tk.Radiobutton(ButtonFrame, text="Reset Mask", value=1,  indicatoron = 0, variable=ResetValue,  command = ResetFunction)
    Reset.place(relx=0.45, rely=0.05, relwidth=0.1, relheight=0.9)

    Previous = tk.Button(ButtonFrame, text="Previous Image", command = _change_image_idx_Previous)
    Previous.place(relx=0.55, rely=0.05, relwidth=0.1, relheight=0.9)

    Next = tk.Button(ButtonFrame, text="Next Image", command = _change_image_idx_Next)
    Next.place(relx=0.65, rely=0.05, relwidth=0.1, relheight=0.9)

    Display_ImgIndex = tk.Label(ButtonFrame, borderwidth=2, relief="groove")
    Display_ImgIndex.place(relx=0.75, rely=0.05, relwidth=0.1, relheight=0.9)

    SaveMask = tk.Button(ButtonFrame, text="Save Mask", command = save_mask)
    SaveMask.place(relx=0.85, rely=0.05, relwidth=0.1, relheight=0.9)

root.mainloop()