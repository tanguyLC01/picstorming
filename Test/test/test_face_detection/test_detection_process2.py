import cv2
from utils_cv.facedetection import detection_process2
import tensorflow as tf
import pickle
import numpy as np

# fonction vérifie que la fonction trace des rectangles autour des visages sur la photo


def test_detection_process2():
    # given
    mask_model = tf.keras.models.load_model(
        "./ResNet50_v2/ResNet50_mask_detector.model")
    face_model = cv2.dnn.readNet(
        "face_detector/deploy.prototxt", "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
    img = cv2.imread(r'.\Test\test\data_images\plusieurs_visages.jpg')
    # when
    detecting_image_process = detection_process2(img, mask_model, face_model)
    # then
    assert np.array_equal(cv2.resize(img, (500, 500)),
                          detecting_image_process) == False


test_detection_process2()
