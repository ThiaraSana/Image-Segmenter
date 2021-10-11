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

IMG = io.imread("/Users/sanaahmed/Documents/GitHub/Image-Segmenter/untitled/train/untitled folder/image.png")
Mask = io.imread("/Users/sanaahmed/Documents/GitHub/Image-Segmenter/untitled/train/train_masks/train/Class1/thumbnail_image.png")
print(radiomics.glcm.RadiomicsGLCM.getAutocorrelationFeatureValue(IMG))
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