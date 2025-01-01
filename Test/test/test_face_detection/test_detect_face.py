import cv2
import numpy as np
from utils_cv.facedetection import detect_face
from utils_cv.image_processing_operations import crop

# fonction qui vérifie que detect_face trace un rectangle autour du visage sur la photo


def test_detect_face():
    # given
    img1 = cv2.imread(
        r'.\Test\test\data_images\mer.jpg')  # image sans visage
    # image avec visage
    img2 = cv2.imread(r'.\Test\test\data_images\Aaron_Eckhart_0001.jpg')
    points = [[(161, 127), (338, 388)]]
    face_model = cv2.dnn.readNet(
        "face_detector/deploy.prototxt", "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
    # when
    point1 = detect_face(img1, face_model)[1]
    point2 = detect_face(img2, face_model)[1]
    # then
    assert point1 == [[]]  # on vérifie qu'aucun visage n'a été détecté
    # on vérifie que le rectangle est bien tracé autour du visage
    assert point2 == points


test_detect_face()
