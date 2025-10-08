import numpy as np
import os
from glob import glob
import cv2

from Config.config import IMAGE_SIZE

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.bilateralFilter(img, 5, 25, 25)
    _, img = cv2.threshold(img, 60, 125, cv2.THRESH_BINARY_INV)
    img = img.flatten() / 255.0
    return img

def load_img(dataset_path):
    images = []
    labels = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue
        for img_path in glob(os.path.join(dataset_path, label, '*.png')):
            try:
                img = preprocess_image(img_path)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    images = np.array(images)
    labels = np.array(labels)
    images = images.astype('float32') / 255.0
    return images, labels