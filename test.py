import numpy as np
from sklearn.cluster import DBSCAN
from skimage import io
import matplotlib.pyplot as plt
import cv2
import imghdr

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

img = io.imread("/Users/sanaahmed/Desktop/untitled/train/image.png")
model = KElbowVisualizer(KMeans(), k=70)
model.fit(img)
model.show()
# img = cv2.imread("/Users/sanaahmed/Desktop/test.png")
# labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# print(imghdr.what("/Users/sanaahmed/Desktop/untitled/train/train_masks/train/Class1/image.png", h=None))
# img = io.imread("/Users/sanaahmed/Desktop/untitled/train/image.png")
# p = np.array(img)
# print(p.shape)
# cellimg = io.imread("/Users/sanaahmed/Desktop/WorkingDirectory/data/image_and_masks/train_imgs/train/0_85104343_z_0.5.png")
# cellp = np.array(cellimg)
# print(cellp.shape)
# n = 0
# while(n<4):
#     labimg = cv2.pyrDown(labimg)
#     n = n+1

# feature_image=np.reshape(img, [-1, 3])
# rows, cols, chs = img.shape
# db = DBSCAN(eps=0.0005, min_samples=1, metric = 'euclidean',algorithm ='auto')
# db.fit(feature_image)
# labels = db.labels_
# clusterpic = np.reshape(labels, [rows, cols])
# plt.imshow(clusterpic)
# plt.show()
