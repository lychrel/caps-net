import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
from glob import glob


X_data = []

d=0
for fn in glob('data/*.jpg'):
    im = cv2.imread(fn)
    new_im = cv2.resize(im, (32, 32))
    cv2.imwrite('resized/im_%d.jpg'%d, new_im)
    imag = cv2.imread ('resized/im_%d.jpg'%d)
    X_data.append(imag)
    d = d + 1

y_data = np.zeros((6,10))
y_data[0][1] = 1
y_data[1][2] = 1
y_data[2][7] = 1
y_data[3][3] = 1
y_data[4][2] = 1
y_data[5][3] = 1

print('Apt_data shape:', np.array(X_data).shape)
print('Label shape: ', y_data.shape)

def rgb2gray(images):
    """Convert images from rbg to grayscale
    """
    greyscale = np.dot(images, [0.2989, 0.5870, 0.1140])
    return np.expand_dims(greyscale, axis=3)

# Transform the images to greyscale
X_u = rgb2gray(X_data).astype(np.float32)

# Create file
h5f = h5py.File('data/apt_num.h5', 'w')

# Store the datasets
h5f.create_dataset('apt_num_dataset', data=X_u)
h5f.create_dataset('apt_num_labels', data=y_data)

# Close the file
h5f.close()
