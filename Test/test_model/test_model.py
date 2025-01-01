import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
from utils_cv.facedetection import crop, detect_face
from utils_cv.image_processing_operations import resize_image


def create_y_train(y_label):
    '''crée une liste de tableaux décrivant si la personne est masquée'''
    res = []
    for ent in y_label:
        if ent == "mask":
            res.append(np.array([1, 0])) # si la personne est masquée
        else:
            res.append(np.array([0, 1])) # si la personne n'est pas masquée
    return np.array(res)


def create_x_train(x_label, model):
    '''crée une liste d'images de visages croppés'''
    res = []
    for name in x_label:
        sample = cv2.imread("./Test/data_test/"+name)
        img, points = detect_face(sample, model)
        sample = crop(img, points)[0]
        sample = resize_image(sample, (224, 224))/255
        sample = sample.reshape(224, 224, 3)
        res.append(sample)
    return np.array(res)


def test_model():
    '''teste la validation du modèle'''
    model = tf.keras.models.load_model(
        ".\ResNet50_v2\ResNet50_mask_detector.model")
    face_model = cv2.dnn.readNet(
    "face_detector/deploy.prototxt", "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
    db = pd.read_csv("./Test/test_model/test_data_info.csv")
    x_test = db["Name"] # nom de la photo
    y_test = db["Type"] # type de la photo : mask ou no_mask
    x_test = create_x_train(x_test, face_model)
    y_test = create_y_train(y_test)
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}')


test_model()
