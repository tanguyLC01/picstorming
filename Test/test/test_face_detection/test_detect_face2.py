import cv2
import numpy as np
from utils_cv.facedetection import detect_face2
from utils_cv.image_processing_operations import crop

# fonction qui vérifie que detect_face2 détecte tous les visages sur la photo


def test_detect_face2():
    # given
    img1 = cv2.imread(
        r'.\Test\test\data_images\mer.jpg')  # image sans visage
    # image avec deux visages
    img2 = cv2.imread(r'.\Test\test\data_images\plusieurs_visages.jpg')
    points = [[(308, 124), (395, 287)], [(144, 83), (232, 242)]]
    face_model = cv2.dnn.readNet(
        "face_detector/deploy.prototxt", "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
    # when
    point1 = detect_face2(img1, face_model)[1]
    point2 = detect_face2(img2, face_model)[1]
    # then
    assert point1 == []
    assert point2 == points


test_detect_face2()
