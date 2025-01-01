import cv2
from utils_cv.facedetection import is_masked
import tensorflow as tf
import pickle

# fonction qui vérifie que la fonction is_masked reconnaît si l'individu sur la photo porte un masque ou non


def test_is_masked():
    # given
    with open(r'.\Test\test\data_images\img_cropped_mask.py', 'rb') as f1:
        img1 = pickle.load(f1)  # image avec masque
    with open(r'.\Test\test\data_images\img_cropped_no_mask.py', 'rb') as f:
        img2 = pickle.load(f)  # image sans masque
    mask_model = tf.keras.models.load_model(
        "./ResNet50_v2/ResNet50_mask_detector.model")
    # then
    assert is_masked(img1, mask_model)[0] == True
    assert is_masked(img2, mask_model)[0] == False


test_is_masked()
