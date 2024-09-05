import numpy as np
import os, sys
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import itertools
import scipy.stats
import tensorflow as tf
from keras import applications, optimizers, Input
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.multiclass import unique_labels

folder = '/home/brunopaiva/MRI_Brain_Tumor/MRI_Brain_Tumor/Train/'
print(folder)

image_width = 128
image_height = 128
channels = 1

train_files = []
i = 0
for mri in ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']:
    onlyfiles = [f for f in os.listdir(os.path.join(folder, mri + '/images'))if os.path.isfile(os.path.join(folder, mri + '/images', f))]
    for _file in onlyfiles:
        train_files.append(_file)

dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels),
                     dtype=np.float32)
y_dataset = []

i = 0
for mri in ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']:
    onlyfiles = [f for f in os.listdir(os.path.join(folder, mri + '/images'))if os.path.isfile(os.path.join(folder, mri + '/images', f))]
    for _file in onlyfiles:
        img_path = os.path.join(folder, mri + '/images', _file)
        img = load_img(img_path, target_size=(image_height, image_width), color_mode='grayscale')
        x = img_to_array(img)
        dataset[i] = x
        mapping = {'Glioma' : 0, 'Meningioma' : 1, 'No Tumor' : 2, 'Pituitary' : 3}
        y_dataset.append(mapping[mri])
        i += 1
        if i == 30000:
            print("%d images to array" % i)
            break
print("All images to array!")

dataset = dataset.astype('float32')
dataset /= 255

classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituiary']

first_image_index = {}

for i, label in enumerate(y_dataset):
    if label not in first_image_index:
        first_image_index[label] = i

num_classes = len(set(y_dataset))
num_images_per_class = 1
num_cols = num_classes
num_rows = num_images_per_class

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))

for i in range(num_classes):
    idx = first_image_index[i]

    pixels = dataset[idx].reshape(image_height, image_width)

    axes[i].imshow(pixels, cmap='gray')
    axes[i].axis('off')

    axes[i].set_title(f'{classes[i]}')

# Exibe a figura
plt.tight_layout()
plt.show()
