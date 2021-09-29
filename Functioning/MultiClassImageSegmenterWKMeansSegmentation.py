import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path 

import numpy as np

from skimage import io
from sklearn.cluster import KMeans

import os
from os import path

import glob

#Global Variables

VALID_IMAGE_TYPES = ['jpeg', 'png', 'bmp', 'gif', 'jpg']
FIGSIZE=(9,7)
OVERLAY_ALPHA=.5
OVERLAY_ERASE = 0
HEIGHT_OF_GUI = 60
WIDTH_OF_GUI = 1500
IMG_IDX = 0
INDICES = None
LINEPROPS = {'color': 'black', 'linewidth': 1, 'alpha': 0.8}
ZOOM_SCALE=1.1
NUMBEROFIMAGES = 0
IMG = []
IMAGE_PATHS = []
PIX = []
MASK_DIR_0 = []
MASK_DIR_1 = []
MASK_DIR_2 = []
MASK_DIR_3 = []
MASK_DIR_4 = []
MASK_PATH_0 = []
MASK_PATH_1 = []
MASK_PATH_2 = []
MASK_PATH_3 = []
MASK_PATH_4 = []
CLASS_MASK_0 = []
CLASS_MASK_1 = []
CLASS_MASK_2 = []
CLASS_MASK_3 = []

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

def Get_Folder():
    global MASK_DIR_0,MASK_DIR_1,MASK_DIR_2,MASK_DIR_3, MASK_DIR_4, IMAGE_PATHS, INDICES, NUMBEROFIMAGES
    img_dir = filedialog.askdirectory()
    if not path.isdir(path.join(img_dir, 'train')):
        raise ValueError(f"{img_dir} must exist and contain the the folder 'train'")
    img_dir = path.join(img_dir, 'train')
    MASK_DIR_0 = path.join(img_dir.rsplit('train_imgs/',1)[0], 'train_masks/train/Class1')
    MASK_DIR_1 = path.join(img_dir.rsplit('train_imgs/',1)[0], 'train_masks/train/Class2')
    MASK_DIR_2 = path.join(img_dir.rsplit('train_imgs/',1)[0], 'train_masks/train/Class3')
    MASK_DIR_3 = path.join(img_dir.rsplit('train_imgs/',1)[0], 'train_masks/train/Class4')
    MASK_DIR_4 = path.join(img_dir.rsplit('train_imgs/',1)[0], 'train_masks/train/MultiClass')
    if not os.path.isdir(MASK_DIR_0):
        os.makedirs(MASK_DIR_0)
    for type_ in VALID_IMAGE_TYPES:
        IMAGE_PATHS += (glob.glob(img_dir.rstrip('/')+f'/*.{type_}'))
        NUMBEROFIMAGES = len(IMAGE_PATHS)
    Show_Image(IMG_IDX)

def Show_Image(IMG_IDX):
    global  DISPLAYED, LASSO, IMG, CLASS_MASK_0, CLASS_MASK_1, CLASS_MASK_2, CLASS_MASK_3, PIX,  MASK_PATH_0, MASK_PATH_1,  MASK_PATH_2, MASK_PATH_3, MASK_PATH_4
    plt.ion()
    shape = None
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.gca()
    IMG = io.imread(IMAGE_PATHS[IMG_IDX])
    img_path = IMAGE_PATHS[IMG_IDX]
    ax.set_title(os.path.basename(img_path))
    MASK_PATH_0 = MASK_DIR_0 + f'/{os.path.basename(img_path)}'
    MASK_PATH_1 = MASK_DIR_1 + f'/{os.path.basename(img_path)}'
    MASK_PATH_2 = MASK_DIR_2 + f'/{os.path.basename(img_path)}'
    MASK_PATH_3 = MASK_DIR_3 + f'/{os.path.basename(img_path)}'
    MASK_PATH_4 = MASK_DIR_4 + f'/{os.path.basename(img_path)}'
    shape = IMG.shape
    pix_x = np.arange(shape[0])
    pix_y = np.arange(shape[1])
    xv, yv = np.meshgrid(pix_y,pix_x)
    PIX = np.vstack( (xv.flatten(), yv.flatten()) ).T
    DISPLAYED = ax.imshow(IMG)

    Display_ImgIndex['text'] = 'Image Number: ' + str(root.counter) + '/' + str(NUMBEROFIMAGES)

    if os.path.exists(MASK_PATH_0):
        CLASS_MASK_0 = io.imread(MASK_PATH_0)
    else:
        CLASS_MASK_0 = np.zeros([shape[0],shape[1]],dtype=np.uint8)

    if os.path.exists(MASK_PATH_1):
        CLASS_MASK_1 = io.imread(MASK_PATH_1)
    else:
        CLASS_MASK_1 = np.zeros([shape[0],shape[1]],dtype=np.uint8)

    if os.path.exists(MASK_PATH_2):
        CLASS_MASK_2 = io.imread(MASK_PATH_2)
    else:
        CLASS_MASK_2 = np.zeros([shape[0],shape[1]],dtype=np.uint8)

    if os.path.exists(MASK_PATH_3):
        CLASS_MASK_3 = io.imread(MASK_PATH_3)
    else:
        CLASS_MASK_3 = np.zeros([shape[0],shape[1]],dtype=np.uint8)

    LASSO = LassoSelector(ax, Render_Lasso, lineprops=LINEPROPS)
    LASSO.set_active(True)
    
    if ZOOM_SCALE is not None:
            disconnect_scroll = zoom_factory(ax, base_scale = ZOOM_SCALE)

# def KMeansSegmentation():
#     fig = plt.figure(figsize=FIGSIZE)
#     ax = fig.gca()
#     reshapedImg = IMG.reshape(IMG.shape[0], IMG.shape[1]) #*IMG.shape[1]
#     kmeans = KMeans(n_clusters=5, random_state=0).fit(reshapedImg)
#     IMG2show = kmeans.cluster_centers_[kmeans.labels_]
#     cluster_IMG = IMG2show.reshape(IMG.shape[0], IMG.shape[1]) #, IMG.shape[2]
#     ax.imshow(cluster_IMG)

def Choose_Color(*args):
    for ActionIndex,ActionType in ActionDictionary.items():
        if ActionType==options_draw.get():
            if ActionIndex == 1:
                for ClassColor,ClassType in ClassDictionary.items():
                    if ClassType==options_class.get():
                        return ClassColor          
            elif ActionIndex == 2: 
                ClassColor = 0
                return ClassColor

def Choose_Class(*args):
    for ClassIndex,ClassName in ClassDictionary.items():
        if ClassName==options_class.get():
                return ClassIndex 

def Render_Lasso(verts):
    LassoPath = Path(verts)
    Color = Choose_Color()
    Class = Choose_Class()
    INDICES = LassoPath.contains_points(PIX, radius=0).reshape(1024,1024) #(450,540)
    Update_Array(INDICES, ResetValue = 0, ClassColor = Color, WhichClass = Class)

def Update_Array(INDICES, ResetValue, ClassColor, WhichClass):

    array = DISPLAYED.get_array().data
    print("test:" + str(array.shape))
    ImageClasses = [['Non-Perfusion Area'], ['Blockage Artefact'], ['High Standard Deviation Artefact'], ['Perfusion Area']]
    if isinstance(ImageClasses, int):
        ImageClasses = np.arange(ImageClasses)
    if len(ImageClasses)<=10:
        colors = 'tab10'
    elif len(ImageClasses)<=20:
        colors = 'tab20'
    else:
        raise ValueError(f'Currently only up to 20 ImageClasses are supported, you tried to use {len(ImageClasses)} ImageClasses')
    colors = np.vstack([[0,0,0], plt.get_cmap(colors)(np.arange(len(ImageClasses)))[:,:3]])

    class_dropdown = ClassColor
    if WhichClass == 1:
        ClassMask = CLASS_MASK_0
    elif WhichClass == 2:
        ClassMask = CLASS_MASK_1
    elif WhichClass == 3:
        ClassMask = CLASS_MASK_2
    elif WhichClass == 4:
        ClassMask = CLASS_MASK_3

    if ResetValue ==1: ## To Reset
        ClassMask[INDICES] = 0
        array[INDICES] = IMG[INDICES]
    elif class_dropdown == 0: ## To Erase
        ClassMask[INDICES] = 0
        c_overlay = colors[ClassMask[INDICES]]*255*OVERLAY_ERASE
        array[INDICES] = (c_overlay + IMG[INDICES]*(1-OVERLAY_ERASE))
    elif INDICES is not None: ## To Draw
        ClassMask[INDICES] = class_dropdown
        c_overlay = colors[ClassMask[INDICES]]*255*OVERLAY_ALPHA
        array[INDICES] = (IMG[INDICES]*(1-OVERLAY_ALPHA)) #c_overlay + ## Problem: c_overlay has 3 channels (rgb) but test image is bmp so no 3 channels

    else:
        idx = ClassMask != 0
        c_overlay = colors[ClassMask[idx]]*255*OVERLAY_ALPHA
        array[idx] = (c_overlay + IMG[idx]*(1-OVERLAY_ALPHA))
    
    DISPLAYED.set_data(array)

def Reset_Mask():
    if (ResetValue.get() ==1):
        Update_Array(INDICES, ResetValue = 1, WhichClass=0)

def Which_Class_To_Save(*args):
    for SavingIndex,ClassType in SavingDictionary.items():
        if ClassType==options_save.get():
            if SavingIndex == 1:
                Save_Mask(CLASS_MASK_0, MASK_PATH_0)
            elif SavingIndex == 2:
                Save_Mask(CLASS_MASK_1, MASK_PATH_1)
            elif SavingIndex == 3:
                Save_Mask(CLASS_MASK_2, MASK_PATH_2)
            elif SavingIndex == 4:
                Save_Mask(CLASS_MASK_3, MASK_PATH_3)
            elif SavingIndex == 5:
                Save_MultiClass_Mask(CLASS_MASK_0, CLASS_MASK_1, CLASS_MASK_2, CLASS_MASK_3, MASK_PATH_4)

def Save_Mask(ClassMask, MaskPath, save_if_no_nonzero=False):
    if (save_if_no_nonzero or np.any(ClassMask != 0)):
        if os.path.splitext(MaskPath)[1] in ['jpg', 'jpeg']:
            io.imsave(MaskPath, ClassMask*255, check_contrast =False, quality=100)
        else:
            io.imsave(MaskPath, ClassMask*255, check_contrast =False)

def Save_MultiClass_Mask(ClassMask1, ClassMask2, ClassMask3, ClassMask4, MaskPath, save_if_no_nonzero=False):
    stack1 = np.add(ClassMask1,ClassMask2)
    stack2 = np.add(ClassMask3, ClassMask4)
    MultiClass_Mask = np.add(stack1, stack2)

    if (save_if_no_nonzero or np.any(MultiClass_Mask != 0)):
        if os.path.splitext(MaskPath)[1] in ['jpg', 'jpeg']:
            io.imsave(MaskPath, MultiClass_Mask*100, check_contrast =False, quality=100)
        else:
            io.imsave(MaskPath, MultiClass_Mask*100, check_contrast =False)

def Next_Image_Index():
    global IMG_IDX
    if IMG_IDX +1 < len(IMAGE_PATHS):
        IMG_IDX += 1
        root.counter += 1
        Display_ImgIndex['text'] = 'Image Number: ' + str(root.counter) + '/' + str(NUMBEROFIMAGES)
        Show_Image(IMG_IDX)
        if IMG_IDX == len(IMAGE_PATHS):
            messagebox.showerror("Error", "There are no more images in folder")

def Previous_Image_Index():
    global IMG_IDX
    if IMG_IDX>=1:
        IMG_IDX -= 1
        root.counter -= 1
        Display_ImgIndex['text'] = 'Image Number: ' + str(root.counter) + '/' + str(NUMBEROFIMAGES)
        Show_Image(IMG_IDX)
        if IMG_IDX == 0:
            messagebox.showerror("Error", "You have reached the beginning of the folder.")
    
#Tkinter Window

if __name__=="__main__":
    root = tk.Tk()
    root.title("Image Segmenter")

    ResetValue = tk.IntVar()
    ClassDictionary={1: 'Non-Perfusion Area', 2: 'Blockage Artefact', 3: 'High Standard Deviation Artefact', 4: 'Perfusion Area'}
    ActionDictionary={1: 'Draw', 2: 'Erase'}
    SavingDictionary={1: 'Non-Perfusion Area Class', 2: 'Blockage Artefact Class', 3: 'High Standard Deviation Artefact Class', 4: 'Perfusion Area Class', 5: 'Multi-Class'}

    canvas = tk.Canvas(root, height=HEIGHT_OF_GUI, width=WIDTH_OF_GUI)
    canvas.pack(side=tk.RIGHT, fill = tk.BOTH, expand=2)

    ButtonFrame = tk.Frame(root)
    ButtonFrame.place(relx=0, rely=0, relwidth=1, relheight=1)
    
    root.counter = 1
    options_class = tk.StringVar(ButtonFrame)
    options_class.set('Select Mask:')
    options_draw = tk.StringVar(ButtonFrame)
    options_draw.set(ActionDictionary[1])
    options_save = tk.StringVar(ButtonFrame)
    options_save.set('Save Mask:')

##Row1
    LoadFolder = tk.Button(ButtonFrame, text="Load Folder", command = Get_Folder)
    LoadFolder.place(relx=0.05, rely=0.05, relwidth=0.06, relheight=0.4)

    ClassDropDown = tk.OptionMenu( ButtonFrame, options_class, *ClassDictionary.values())
    ClassDropDown.place(relx=0.11, rely=0.05, relwidth=0.2, relheight=0.4)

    Action = tk.OptionMenu(ButtonFrame, options_draw, *ActionDictionary.values())
    Action.place(relx=0.31, rely=0.05, relwidth=0.06, relheight=0.4)
    
    Reset = tk.Radiobutton(ButtonFrame, text="Reset Mask", value=1,  indicatoron = 0, variable=ResetValue,  command = Reset_Mask)
    Reset.place(relx=0.37, rely=0.05, relwidth=0.06, relheight=0.4)

    PreviousImage = tk.Button(ButtonFrame, text="Previous", command = Previous_Image_Index)
    PreviousImage.place(relx=0.43, rely=0.05, relwidth=0.06, relheight=0.4)

    NextImage = tk.Button(ButtonFrame, text="Next", command = Next_Image_Index)
    NextImage.place(relx=0.49, rely=0.05, relwidth=0.06, relheight=0.4)

    Display_ImgIndex = tk.Label(ButtonFrame, borderwidth=2, relief="groove")
    Display_ImgIndex.place(relx=0.55, rely=0.1, relwidth=0.15, relheight=0.4)

    ChooseMaskToSave = tk.OptionMenu(ButtonFrame, options_save, *SavingDictionary.values())
    ChooseMaskToSave.place(relx=0.7, rely=0.05, relwidth=0.2, relheight=0.4)
    
    SaveMask = tk.Button(ButtonFrame, text="Save Image", command = Which_Class_To_Save)
    SaveMask.place(relx=0.9, rely=0.05, relwidth=0.06, relheight=0.4)

##Row2
    # AutoSegmentation = tk.Button(ButtonFrame, text="KMeans", command = KMeansSegmentation)
    # AutoSegmentation.place(relx=0.05, rely=0.45, relwidth=0.1, relheight=0.4)

root.mainloop()