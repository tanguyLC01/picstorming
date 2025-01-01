import tensorflow as tf
from utils_cv.image_processing_operations import resize_image, crop
import numpy as np
import cv2
from Model.model import use_model
import time
import pickle
SIZE_IMAGE = 500


def get_mask_status():
    return NO_MASK_DETECTED


"""
# load Models for face detection and face mask detection
mask_model = tf.keras.models.load_model(
    "./ResNet50_v2/ResNet50_mask_detector.model")
face_model = cv2.dnn.readNet(
    "face_detector/deploy.prototxt", "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
"""

############ Functions for one face detection #############


def detect_face(img, face_model):
    '''détecte un visage, renvoie l'image et les coordonnées du rectangle'''
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))  # preprocess pour normaliser la data
    face_model.setInput(blob)
    detections = face_model.forward()
    top_detections = []
    for i in range(len(detections[0, 0, :, :])):
        if detections[0, 0, i, 2] >= 0.7:
            top_detections.append(detections[0, 0, i, :])
    if top_detections == []:
        return (img, [top_detections])
    # coordonnées pour le rectangle
    x0, y0, x1, y1 = list(
        map(int, top_detections[0][3:7]*SIZE_IMAGE*np.ones(4)))
    img = resize_image(img, (SIZE_IMAGE, SIZE_IMAGE))
    return (img, [[(x0, y0), (x1, y1)]])


def is_masked(img_cropped, mask_model):
    '''retourne un booléen pour la présence de masque et la probabilité associée'''
    prob = use_model(img_cropped, mask_model)
    if prob[0][0] > prob[0][1]:
        return (True, prob[0][0])  # visage masqué
    return (False, prob[0][1])  # visage non masqué


def detection_process(img, mask_model, face_model):
    '''trace un rectangle autour du visage: vert si masqué, rouge sinon'''
    img = resize_image(img, (SIZE_IMAGE, SIZE_IMAGE))
    face, points = detect_face(img, face_model)
    mask_detected, prob = is_masked(crop(face, points)[0], mask_model)
    if mask_detected:
        color = (0, 255, 0)  # vert
        result = 'mask'
    else:
        color = (0, 0, 255)  # rouge
        result = 'no mask'
    [[(x0, y0), (x1, y1)]] = points
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    cv2.putText(img, result, (x0, y0-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(img, str(round(prob*100, 2))+'%', (x0, y0-25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
    return img

####### Functions for detecting multiple faces ###########


def detect_face2(img, face_model):
    '''détecte les visages et les coordonnées des rectangles dans une liste'''
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_model.setInput(blob)
    detections = face_model.forward()
    # un visage détecté par le modèle est réellement existant si sa probabilité de présence est supérieure à 0.7
    top_detections = []
    for i in range(0, len(detections[0, 0, :, :])):
        # le 2 est l'index de la probabilité de présence d'un visage
        if detections[0, 0, i, 2] >= 0.7:
            top_detections.append(detections[0, 0, i, :])
    points = []
    for i in range(len(top_detections)):
        x0, y0, x1, y1 = list(
            map(int, top_detections[i][3:7]*SIZE_IMAGE*np.ones(4)))
        points.append([(x0, y0), (x1, y1)])
    return (img, points)


def is_masked2(faces_cropped, mask_model):
    '''retourne une liste de booléen pour la présence de masque et une liste pour la probabilité associée'''
    is_masked = []
    prob_model = []
    # on applique le modèle à chaque image de visage cropped
    for i in range(len(faces_cropped)):
        prob_model.append(
            use_model(faces_cropped[i], mask_model).reshape(-1))
    prob = []
    for i in range(len(prob_model)):
        mask, no_mask = prob_model[i]
        is_masked.append((mask > no_mask))  # ajoute un booléen à la liste
        # ajoute la probabilité associée à la liste
        prob.append(max(mask, no_mask))
    return is_masked, prob


def detection_process2(img, mask_model, face_model):
    '''trace un rectangle autour des visages: vert si masqué, rouge sinon'''
    img = resize_image(img, (SIZE_IMAGE, SIZE_IMAGE))
    img, points = detect_face2(img, face_model)
    mask_detected, prob = is_masked2(np.array(crop(img, points)), mask_model)
    colors = []  # crée une liste regroupant les couleurs de chaque rectangle
    # crée une liste regroupant les caractéristiques de chaque visage (mask ou no mask)
    result = []
    for i in range(len(mask_detected)):
        mask = mask_detected[i]
        if mask:
            colors.append((0, 255, 0))
            result.append('mask')
        else:
            colors.append((0, 0, 255))
            result.append('no mask')
    global NO_MASK_DETECTED
    # False si il y a une personne sans masque
    NO_MASK_DETECTED = all(mask_detected)
    for (index, [(x0, y0), (x1, y1)]) in enumerate(points):
        # trace le rectangle ( 2 est l'épaisseur)
        cv2.rectangle(img, (x0, y0), (x1, y1), colors[index], 2)
        cv2.putText(img, result[index], (x0, y0-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[index], 2)  # écrit mask ou no mask
        cv2.putText(img, str(round(prob[index]*100, 2))+'%', (x0, y0-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[index], 2)  # écrit la probabilité associée
    return img
