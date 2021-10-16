import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path 

import numpy as np

import csv

from skimage import io
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import radiomics
from radiomics import featureextractor

import SimpleITK as sitk

import os
from os import path

import glob

#Global Variables

VALID_IMAGE_TYPES = ['jpeg', 'png', 'bmp', 'gif', 'jpg']
FIGSIZE=(9,7)
OVERLAY_ALPHA=.9
OVERLAY_ERASE = 0
HEIGHT_OF_GUI = 60
WIDTH_OF_GUI = 1500
IMG_IDX = 0
INDICES = None
LINEPROPS = {'color': 'black', 'linewidth': 1, 'alpha': 0.8}
ZOOM_SCALE=1.1
NUMBEROFIMAGES = 0
IMAGE_DIRECTORY = []
IMG = []
IMAGE_PATHS = []
PIX = []

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
    global NONPERFUSION_GREYSCALE, BLOCKAGE_GREYSCALE, HIGHSD_GREYSCALE, PERFUSION_GREYSCALE, MULTICLASS_GREYSCALE, NONPERFUSION_BINARY, BLOCKAGE_BINARY, HIGHSD_BINARY, PERFUSION_BINARY, NONPERFUSION_TEXTURE, BLOCKAGE_TEXTURE, HIGHSD_TEXTURE, PERFUSION_TEXTURE, IMAGE_PATHS, INDICES, NUMBEROFIMAGES, IMAGE_DIRECTORY
    Working_Directory = filedialog.askdirectory()
    if not path.isdir(path.join(Working_Directory, 'Images')):
        raise ValueError(f"{Working_Directory} must exist and contain the the folder 'Images'")
    IMAGE_DIRECTORY = path.join(Working_Directory, 'Images')

    Mask_Directory_Binary = path.join(Working_Directory, 'Masks/Binary_Masks')
    Mask_Directory_GreyScale = path.join(Working_Directory, 'Masks/GreyScale_Masks')

    NONPERFUSION_BINARY = path.join(Mask_Directory_Binary, "NonPerfusion_Binary")
    NONPERFUSION_GREYSCALE = path.join(Mask_Directory_GreyScale, "NonPerfusion_GreyScale")

    BLOCKAGE_BINARY = path.join(Mask_Directory_Binary, "Blockage_Binary")
    BLOCKAGE_GREYSCALE = path.join(Mask_Directory_GreyScale, "Blockage_GreyScale")

    HIGHSD_BINARY = path.join(Mask_Directory_Binary, "HighSD_Binary")
    HIGHSD_GREYSCALE = path.join(Mask_Directory_GreyScale, "HighSD_GreyScale")

    PERFUSION_BINARY = path.join(Mask_Directory_Binary, "Perfusion_Binary")
    PERFUSION_GREYSCALE = path.join(Mask_Directory_GreyScale, "Perfusion_GreyScale")

    MULTICLASS_GREYSCALE = path.join(Mask_Directory_GreyScale, "MultiClass_GreyScale")

    if not os.path.isdir(NONPERFUSION_BINARY):
        os.makedirs(NONPERFUSION_BINARY)
    if not os.path.isdir(NONPERFUSION_GREYSCALE):
        os.makedirs(NONPERFUSION_GREYSCALE)
    if not os.path.isdir(BLOCKAGE_BINARY):
        os.makedirs(BLOCKAGE_BINARY)
    if not os.path.isdir(BLOCKAGE_GREYSCALE):
        os.makedirs(BLOCKAGE_GREYSCALE)
    if not os.path.isdir(HIGHSD_BINARY):
        os.makedirs(HIGHSD_BINARY)
    if not os.path.isdir(HIGHSD_GREYSCALE):
        os.makedirs(HIGHSD_GREYSCALE)
    if not os.path.isdir(PERFUSION_BINARY):
        os.makedirs(PERFUSION_BINARY)
    if not os.path.isdir(PERFUSION_GREYSCALE):
        os.makedirs(PERFUSION_GREYSCALE)
    if not os.path.isdir(MULTICLASS_GREYSCALE):
        os.makedirs(MULTICLASS_GREYSCALE)

    for type_ in VALID_IMAGE_TYPES:
        IMAGE_PATHS += (glob.glob(IMAGE_DIRECTORY.rstrip('/')+f'/*.{type_}'))
        NUMBEROFIMAGES = len(IMAGE_PATHS)
    
    TextureAnalysis_Directory = path.join(Working_Directory, 'TexturalFeatures')

    NONPERFUSION_TEXTURE = path.join(TextureAnalysis_Directory, "NonPerfusion_Texture")
    BLOCKAGE_TEXTURE = path.join(TextureAnalysis_Directory, "Blockage_Texture")
    HIGHSD_TEXTURE = path.join(TextureAnalysis_Directory, "HighSD_Texture")
    PERFUSION_TEXTURE = path.join(TextureAnalysis_Directory, "Perfusion_Texture")

    if not os.path.isdir(NONPERFUSION_TEXTURE):
        os.makedirs(NONPERFUSION_TEXTURE)
    if not os.path.isdir(BLOCKAGE_TEXTURE):
        os.makedirs(BLOCKAGE_TEXTURE)
    if not os.path.isdir(HIGHSD_TEXTURE):
        os.makedirs(HIGHSD_TEXTURE)
    if not os.path.isdir(PERFUSION_TEXTURE):
        os.makedirs(PERFUSION_TEXTURE)

    Show_Image(IMG_IDX)

def Show_Image(IMG_IDX):
    global  DISPLAYED, LASSO, IMG, PIX,  NONPERFUSION_GREYSCALE_MASKPATH, BLOCKAGE_GREYSCALE_MASKPATH,  HIGHSD_GREYSCALE_MASKPATH, PERFUSION_GREYSCALE_MASKPATH, MULTICLASS_GREYSCALE_MASKPATH, NONPERFUSION_GREYSCALE_MASK, BLOCKAGE_GREYSCALE_MASK, HIGHSD_GREYSCALE_MASK, PERFUSION_GREYSCALE_MASK,  NONPERFUSION_BINARY_MASKPATH, BLOCKAGE_BINARY_MASKPATH,  HIGHSD_BINARY_MASKPATH, PERFUSION_BINARY_MASKPATH, MULTICLASS_BINARY_MASKPATH, NONPERFUSION_BINARY_MASK, BLOCKAGE_BINARY_MASK, HIGHSD_BINARY_MASK, PERFUSION_BINARY_MASK
    plt.ion()
    shape = None
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.gca()
    IMG = io.imread(IMAGE_PATHS[IMG_IDX])
    img_path = IMAGE_PATHS[IMG_IDX]
    ax.set_title(os.path.basename(img_path))

    shape = IMG.shape
    pix_x = np.arange(shape[0])
    pix_y = np.arange(shape[1])
    xv, yv = np.meshgrid(pix_y,pix_x)
    PIX = np.vstack( (xv.flatten(), yv.flatten()) ).T
    DISPLAYED = ax.imshow(IMG)
    
    Display_ImgIndex['text'] = 'Image Number: ' + str(root.counter) + '/' + str(NUMBEROFIMAGES)

    NONPERFUSION_GREYSCALE_MASKPATH = NONPERFUSION_GREYSCALE + f'/{os.path.basename(img_path)}'
    BLOCKAGE_GREYSCALE_MASKPATH = BLOCKAGE_GREYSCALE + f'/{os.path.basename(img_path)}'
    HIGHSD_GREYSCALE_MASKPATH = HIGHSD_GREYSCALE + f'/{os.path.basename(img_path)}'
    PERFUSION_GREYSCALE_MASKPATH = PERFUSION_GREYSCALE + f'/{os.path.basename(img_path)}'
    MULTICLASS_GREYSCALE_MASKPATH = MULTICLASS_GREYSCALE + f'/{os.path.basename(img_path)}'

    if os.path.exists(NONPERFUSION_GREYSCALE_MASKPATH):
        NONPERFUSION_GREYSCALE_MASK = io.imread(NONPERFUSION_GREYSCALE_MASKPATH)
    else:
        NONPERFUSION_GREYSCALE_MASK = np.zeros([shape[0],shape[1]],dtype=np.uint8)

    if os.path.exists(BLOCKAGE_GREYSCALE_MASKPATH):
        BLOCKAGE_GREYSCALE_MASK = io.imread(BLOCKAGE_GREYSCALE_MASKPATH)
    else:
        BLOCKAGE_GREYSCALE_MASK = np.zeros([shape[0],shape[1]],dtype=np.uint8)

    if os.path.exists(HIGHSD_GREYSCALE_MASKPATH):
        HIGHSD_GREYSCALE_MASK = io.imread(HIGHSD_GREYSCALE_MASKPATH)
    else:
        HIGHSD_GREYSCALE_MASK = np.zeros([shape[0],shape[1]],dtype=np.uint8)

    if os.path.exists(PERFUSION_GREYSCALE_MASKPATH):
        PERFUSION_GREYSCALE_MASK = io.imread(PERFUSION_GREYSCALE_MASKPATH)
    else:
        PERFUSION_GREYSCALE_MASK = np.zeros([shape[0],shape[1]],dtype=np.uint8)

    NONPERFUSION_BINARY_MASKPATH = NONPERFUSION_BINARY + f'/{os.path.basename(img_path)}'
    BLOCKAGE_BINARY_MASKPATH = BLOCKAGE_BINARY + f'/{os.path.basename(img_path)}'
    HIGHSD_BINARY_MASKPATH = HIGHSD_BINARY + f'/{os.path.basename(img_path)}'
    PERFUSION_BINARY_MASKPATH = PERFUSION_BINARY + f'/{os.path.basename(img_path)}'

    if os.path.exists(NONPERFUSION_BINARY_MASKPATH):
        NONPERFUSION_BINARY_MASK = io.imread(NONPERFUSION_BINARY_MASKPATH)
    else:
        NONPERFUSION_BINARY_MASK = np.zeros([shape[0],shape[1]],dtype=np.uint8)

    if os.path.exists(BLOCKAGE_BINARY_MASKPATH):
        BLOCKAGE_BINARY_MASK = io.imread(BLOCKAGE_BINARY_MASKPATH)
    else:
        BLOCKAGE_BINARY_MASK = np.zeros([shape[0],shape[1]],dtype=np.uint8)

    if os.path.exists(HIGHSD_BINARY_MASKPATH):
        HIGHSD_BINARY_MASK = io.imread(HIGHSD_BINARY_MASKPATH)
    else:
        HIGHSD_BINARY_MASK = np.zeros([shape[0],shape[1]],dtype=np.uint8)

    if os.path.exists(PERFUSION_BINARY_MASKPATH):
        PERFUSION_BINARY_MASK = io.imread(PERFUSION_BINARY_MASKPATH)
    else:
        PERFUSION_BINARY_MASK = np.zeros([shape[0],shape[1]],dtype=np.uint8)

    LASSO = LassoSelector(ax, Render_Lasso, lineprops=LINEPROPS)
    LASSO.set_active(True)
    
    if ZOOM_SCALE is not None:
            disconnect_scroll = zoom_factory(ax, base_scale = ZOOM_SCALE)

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

    class_dropdown = ClassColor
    if WhichClass == 1:
        ClassMask_GreyScale = NONPERFUSION_GREYSCALE_MASK
        ClassMask_Binary = NONPERFUSION_BINARY_MASK
    elif WhichClass == 2:
        ClassMask_GreyScale = BLOCKAGE_GREYSCALE_MASK
        ClassMask_Binary = BLOCKAGE_BINARY_MASK
    elif WhichClass == 3:
        ClassMask_GreyScale = HIGHSD_GREYSCALE_MASK
        ClassMask_Binary = HIGHSD_BINARY_MASK
    elif WhichClass == 4:
        ClassMask_GreyScale = PERFUSION_GREYSCALE_MASK
        ClassMask_Binary = PERFUSION_BINARY_MASK
    
    if ResetValue ==1: ## To Reset
        ClassMask_GreyScale[INDICES] = 0
        array[INDICES] = IMG[INDICES]
    elif class_dropdown == 0: ## To Erase
        ClassMask_GreyScale[INDICES] = 0
        c_overlay = colors[ClassMask_GreyScale[INDICES]]*255*OVERLAY_ERASE
        array[INDICES] = (IMG[INDICES]*(1-OVERLAY_ERASE)) #c_overlay + 
    elif INDICES is not None: ## To Draw
        ClassMask_GreyScale[INDICES] = class_dropdown
        ClassMask_Binary[INDICES] = 1
        c_overlay = colors[ClassMask_GreyScale[INDICES]]*255*OVERLAY_ALPHA 
        array[INDICES] = (IMG[INDICES]*(1-OVERLAY_ALPHA)) #c_overlay + ## Problem: c_overlay has 3 channels (rgb) but test image is bmp so no 3 channels
    else:
        idx = ClassMask_GreyScale != 0
        c_overlay = colors[ClassMask_GreyScale[idx]]*255*OVERLAY_ALPHA
        array[idx] = (IMG[idx]*(1-OVERLAY_ALPHA)) #c_overlay +
    
    DISPLAYED.set_data(array)

def Reset_Mask():
    if (ResetValue.get() ==1):
        Update_Array(INDICES, ResetValue = 1, WhichClass=0)

def Which_Class_To_Save(*args):
    for SavingIndex,ClassType in SavingDictionary.items():
        if ClassType==options_save.get():
            if SavingIndex == 1:
                Save_Mask(NONPERFUSION_GREYSCALE_MASK, NONPERFUSION_GREYSCALE_MASKPATH, NONPERFUSION_BINARY_MASK, NONPERFUSION_BINARY_MASKPATH)

            elif SavingIndex == 2:
                Save_Mask(BLOCKAGE_GREYSCALE_MASK, BLOCKAGE_GREYSCALE_MASKPATH, BLOCKAGE_BINARY_MASK, BLOCKAGE_BINARY_MASKPATH)

            elif SavingIndex == 3:
                Save_Mask(HIGHSD_GREYSCALE_MASK, HIGHSD_GREYSCALE_MASKPATH, HIGHSD_BINARY_MASK, HIGHSD_BINARY_MASKPATH)

            elif SavingIndex == 4:
                Save_Mask(PERFUSION_GREYSCALE_MASK, PERFUSION_GREYSCALE_MASKPATH, PERFUSION_BINARY_MASK, PERFUSION_BINARY_MASKPATH)

            elif SavingIndex == 5:
                Save_MultiClass_Mask(NONPERFUSION_GREYSCALE_MASK, BLOCKAGE_GREYSCALE_MASK, HIGHSD_GREYSCALE_MASK, PERFUSION_GREYSCALE_MASK, MULTICLASS_GREYSCALE_MASKPATH)

def Save_Mask(ClassMask_GreyScale, MaskPath_GreyScale, ClassMask_Binary, MaskPath_Binary,save_if_no_nonzero=False):
    if (save_if_no_nonzero or np.any(ClassMask_GreyScale != 0)):
        if os.path.splitext(MaskPath_GreyScale)[1] in ['jpg', 'jpeg']:
            io.imsave(MaskPath_GreyScale, ClassMask_GreyScale*255, check_contrast =False, quality=100)
            io.imsave(MaskPath_Binary, ClassMask_Binary, check_contrast =False)
        else:
            io.imsave(MaskPath_GreyScale, ClassMask_GreyScale*255, check_contrast =False)
            io.imsave(MaskPath_Binary, ClassMask_Binary, check_contrast =False)

def Save_MultiClass_Mask(ClassMask_GreyScale1, ClassMask_GreyScale2, ClassMask_GreyScale3, ClassMask_GreyScale4, MaskPath_GreyScale, save_if_no_nonzero=False):
    stack1 = np.add(ClassMask_GreyScale1,ClassMask_GreyScale2)
    stack2 = np.add(ClassMask_GreyScale3, ClassMask_GreyScale4)
    MultiClass_Mask = np.add(stack1, stack2)

    if (save_if_no_nonzero or np.any(MultiClass_Mask != 0)):
        if os.path.splitext(MaskPath_GreyScale)[1] in ['jpg', 'jpeg']:
            io.imsave(MaskPath_GreyScale, MultiClass_Mask*100, check_contrast =False, quality=100)
        else:
            io.imsave(MaskPath_GreyScale, MultiClass_Mask*100, check_contrast =False)

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

def KMeansSegmentation():...

def DBScanSegmentation():...

def Which_TextureFeature_ToGenerate():
    for ClassIndex,ClassType in TextureDictionary.items():
        if ClassType==options_TextureObject.get():
            if ClassIndex == 1:
                GetPyradiomicsInput(IMAGE_DIRECTORY, NONPERFUSION_BINARY, NONPERFUSION_TEXTURE)
            elif ClassIndex == 2:
                GetPyradiomicsInput(IMAGE_DIRECTORY, BLOCKAGE_BINARY, BLOCKAGE_TEXTURE)
            elif ClassIndex == 3:
                GetPyradiomicsInput(IMAGE_DIRECTORY, HIGHSD_BINARY, HIGHSD_TEXTURE)
            elif ClassIndex == 4:
                GetPyradiomicsInput(IMAGE_DIRECTORY, PERFUSION_BINARY, PERFUSION_TEXTURE)
    
def GetPyradiomicsInput(ImagePath, MaskPath, SavingPath):
    ImageFiles = os.listdir(ImagePath)
    for i in range(0, len(ImageFiles)):
        if ImageFiles[i] == ".DS_Store":
            i+=1
        else:
            ImageFilePath = ImagePath + '/' + ImageFiles[i]
            print(ImageFilePath)
            MaskFilePath = MaskPath + '/'  + ImageFiles[i]
            print(MaskFilePath)
            Image = sitk.ReadImage(ImageFilePath, sitk.sitkInt8)
            Mask = sitk.ReadImage(MaskFilePath, sitk.sitkInt8)
            PyRadiomicsExtraction(Image, Mask, ImageFiles[i], SavingPath)
            i +=1
    
def PyRadiomicsExtraction(Image, Mask, SavingName, SavingPath):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    result = extractor.execute(Image, Mask)

    keys, values = [], []
    for key, value in result.items():
        keys.append(key)
        print(key)
        values.append(value)
        print(value)
    SavingName = SavingName[:-4]
    SavingName = SavingName + ".csv"
    ResultsFile = SavingPath + '/' + SavingName
    with open(ResultsFile, "w") as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerow(keys)
        csvwriter.writerow(values)

#Tkinter Window

if __name__=="__main__":
    root = tk.Tk()
    root.title("Image Segmenter")

    ResetValue = tk.IntVar()
    ClassDictionary={1: 'Non-Perfusion Area', 2: 'Blockage Artefact', 3: 'High Standard Deviation Artefact', 4: 'Perfusion Area'}
    ActionDictionary={1: 'Draw', 2: 'Erase'}
    SavingDictionary={1: 'Non-Perfusion Area Class', 2: 'Blockage Artefact Class', 3: 'High Standard Deviation Artefact Class', 4: 'Perfusion Area Class', 5: 'Multi-Class'}
    TextureDictionary={1: 'Non-Perfusion Area Texture', 2: 'Blockage Artefact Texture', 3: 'High Standard Deviation Artefact Texture', 4: 'Perfusion Area Texture'}
    
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
    options_TextureObject = tk.StringVar(ButtonFrame)
    options_TextureObject.set('Generate Texture Features of:')

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
    
    SaveMask = tk.Button(ButtonFrame, text="Save Mask", command = Which_Class_To_Save)
    SaveMask.place(relx=0.9, rely=0.05, relwidth=0.06, relheight=0.4)

##Row2 - ML Segmentation
    KMeans_Segmentation = tk.Button(ButtonFrame, text="KMeans", command = KMeansSegmentation)
    KMeans_Segmentation.place(relx=0.05, rely=0.45, relwidth=0.1, relheight=0.4)

    DBScan_Segmentation = tk.Button(ButtonFrame, text="DB Scan", command = DBScanSegmentation)
    DBScan_Segmentation.place(relx=0.15, rely=0.45, relwidth=0.1, relheight=0.4)

##Row2 - Feature Classes
    ClassDropDown_TexturalFeatures = tk.OptionMenu( ButtonFrame, options_TextureObject, *TextureDictionary.values())
    ClassDropDown_TexturalFeatures.place(relx=0.25, rely=0.45, relwidth=0.2, relheight=0.4)

    ExtractTexturalFeatures = tk.Button(ButtonFrame, text="Extract Textural Features", command = Which_TextureFeature_ToGenerate)
    ExtractTexturalFeatures.place(relx=0.45, rely=0.45, relwidth=0.3, relheight=0.4)

#     Histogram_Generator = tk.Button(ButtonFrame, text="Histogram", command = HistogramGenerator)
#     Histogram_Generator.place(relx=0.65, rely=0.45, relwidth=0.1, relheight=0.4)

# ##Row2 - Image Filter
#     LoG_Filter = tk.Button(ButtonFrame, text="LoG Filter", command = LoGFilter)
#     LoG_Filter.place(relx=0.75, rely=0.45, relwidth=0.1, relheight=0.4)

#     Wavelet_Transform = tk.Button(ButtonFrame, text="Wavelet", command = WaveletTransformation)
#     Wavelet_Transform.place(relx=0.85, rely=0.45, relwidth=0.1, relheight=0.4)

root.mainloop()