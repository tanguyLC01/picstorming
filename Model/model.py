import tensorflow as tf
import cv2
from utils_cv.image_processing_operations import resize_image, crop
import pickle
import os
import numpy as np

### fonction pour détecter si un visage a un masque:

def process_image(image):
    '''redimensionne l'image à traiter'''
    image = resize_image(image, (224, 224))/255 # normalisation
    image = image.reshape(1, 224, 224, 3)
    return image

def use_model(image, model):
    '''applique le modèle qui renvoie un couple de probabilité :
    [0]=probabilité qu'il y ait un masque
    [1]=probabilité qu'il n'y ait pas de masque'''
    image = process_image(image)
    return model.predict(image)
