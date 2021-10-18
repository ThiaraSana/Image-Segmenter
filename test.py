import skimage
from skimage import io

import imghdr

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path 

import numpy as np

from skimage import io
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

from skimage.feature import greycomatrix, greycoprops

import radiomics
from radiomics import glcm
from radiomics import featureextractor

import os
from os import path

import glob
import SimpleITK as sitk

# IMG = io.imread("/Users/sanaahmed/Documents/GitHub/Image-Segmenter/untitled/train/untitled folder/image.png")
# Mask = io.imread("/Users/sanaahmed/Documents/GitHub/Image-Segmenter/untitled/train/train_masks/train/Class1/thumbnail_image.png")
# print(radiomics.glcm.RadiomicsGLCM.getAutocorrelationFeatureValue(IMG))
# Define the testcase name
# testCase = 'brain1'

# # Get the relative path to pyradiomics\data
# # os.cwd() returns the current working directory
# # ".." points to the parent directory: \pyradiomics\bin\Notebooks\..\ is equal to \pyradiomics\bin\
# # Move up 2 directories (i.e. go to \pyradiomics\) and then move into \pyradiomics\data
# dataDir = os.path.join(os.getcwd(), "..", "..", "data")
# print("dataDir, relative path:", dataDir)
# print("dataDir, absolute path:", os.path.abspath(dataDir))

# # Store the file paths of our testing image and label map into two variables
# imagePath = os.path.join(dataDir, testCase + "_image.nrrd")
# labelPath = os.path.join(dataDir, testCase + "_label.nrrd")

# # Additonally, store the location of the example parameter file, stored in \pyradiomics\bin
# paramPath = os.path.join(os.getcwd(), "..", "Params.yaml")
# print("Parameter file, absolute path:", os.path.abspath(paramPath))
# extractor = featureextractor.RadiomicsFeatureExtractor()
# print(extractor)
# model = KElbowVisualizer(KMeans(), k=70)
# model.fit(img)
# model.show()
# IMG = IMG.reshape(-1,3)
# print(imghdr.what("/Users/sanaahmed/Documents/GitHub/Image-Segmenter/untitled/train/train_masks/train/Class1/thumbnail_image.png", h=None))

#     # Compute DBSCAN
# db = DBSCAN(eps=0.3, min_samples=10).fit(IMG)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_

#     # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)

# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)

# image = sitk.ReadImage("/Users/sanaahmed/Documents/GitHub/Image-Segmenter/untitled/train/thumbnail_image.png", sitk.sitkInt8)
# mask = sitk.ReadImage("/Users/sanaahmed/Documents/GitHub/Image-Segmenter/untitled/train/train_masks/train/Class1/thumbnail_image.png", sitk.sitkInt8)
# extractor = featureextractor.RadiomicsFeatureExtractor()
# result = extractor.execute(image, mask)
# keys, values = [], []

# for key, value in result.items():
#     keys.append(key)
#     values.append(value)
#     print(key, value)
# ImagePaths = []
# ImagePath = "/Users/sanaahmed/Documents/GitHub/Image-Segmenter/untitled/images/"
# ImageFiles = os.listdir(ImagePath)
# for i in range(0, len(ImageFiles)):
#     IF = ImagePath + ImageFiles[i] 
#     ImagePaths.append(IF)
#     i += 1
# print(ImagePaths)
# import csv
# image = sitk.ReadImage("/Users/sanaahmed/Documents/GitHub/Image-Segmenter/untitled/images/thumbnail_image.png", sitk.sitkInt8)
# mask = sitk.ReadImage("/Users/sanaahmed/Documents/GitHub/Image-Segmenter/untitled/Masks/Binary_Masks/NonPerfusion_Binary/thumbnail_image.png", sitk.sitkInt8)
# extractor = featureextractor.RadiomicsFeatureExtractor()
# result = extractor.execute(image,mask)
# # print(result)
# # result = extractor.execute(InputFilePath_Image, InputFilePath_Mask)
# keys, values = [], []
# for key, value in result.items():
#     keys.append(key)
#     values.append(value)
# with open("frequencies.csv", "w") as outfile:
#     csvwriter = csv.writer(outfile)
#     csvwriter.writerow(keys)
#     csvwriter.writerow(values)
image = skimage.io.imread("/Users/sanaahmed/Documents/GitHub/Image-Segmenter/WorkingDirectory_IS/Images/thumbnail_image.png")
histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))
print(bin_edges)
