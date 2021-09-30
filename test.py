from skimage import io

import imghdr

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

img = io.imread("/Users/sanaahmed/Desktop/untitled/train/image.png")
model = KElbowVisualizer(KMeans(), k=70)
model.fit(img)
model.show()

print(imghdr.what("/Users/sanaahmed/Desktop/untitled/train/train_masks/train/Class1/image.png", h=None))

