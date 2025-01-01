import cv2
from utils_cv.facedetection import is_masked2
import tensorflow as tf
import pickle

# fonction qui vérifie que la fonction is_masked reconnaît si les individus sur la photo portent un masque ou non


def test_is_masked2():
    # given
    with open(r'.\Test\test\data_images\img_cropped_plusieurs_visages.py', 'rb') as f1:
        # image avec une personne masqué et une personne sans maque
        img = pickle.load(f1)
    mask_model = tf.keras.models.load_model(
        "./ResNet50_v2/ResNet50_mask_detector.model")
    # then
    assert is_masked2(img, mask_model)[0][0] == True
    assert is_masked2(img, mask_model)[0][1] == False


test_is_masked2()
