import numpy as np
from sklearn.cluster import DBSCAN
from skimage import io
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("/Users/sanaahmed/Desktop/test.png")
labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

n = 0
while(n<4):
    labimg = cv2.pyrDown(labimg)
    n = n+1

feature_image=np.reshape(img, [-1, 3])
rows, cols, chs = img.shape
db = DBSCAN(eps=0.0005, min_samples=1, metric = 'euclidean',algorithm ='auto')
db.fit(feature_image)
labels = db.labels_
clusterpic = np.reshape(labels, [rows, cols])
plt.imshow(clusterpic)
plt.show()
