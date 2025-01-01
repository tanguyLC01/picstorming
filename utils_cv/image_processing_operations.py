import numpy as np
import cv2


def resize_image(img, dim):
    '''redimensionne l'image'''
    # dim est un tuple de (x,y)
    resized_img = cv2.resize(img, dim)
    return resized_img


def rotate_image(img, degree):
    '''fait pivoter l'image'''
    (rows, cols, d) = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), degree, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def smoothing_image(img, kernel, dev):
    '''floute l'image'''
    # kernel=(width,height) width and height of the kernel which should be positive and odd
    # dev standard deviation in the X and Y directions. If only sigmaX is specified, sigmaY is taken as equal to sigmaX. If both are given as zeros, they are calculated from the kernel size
    # mettre dev=0
    blur = cv2.GaussianBlur(img, kernel, dev)
    # pour afficher:
    # cv2.imshow('image',blur)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return blur


def draw_rectangle(img, tlcorner, brcorner, color, line_thickness):
    '''dessine un rectangle'''
    cv2.rectangle(img, tlcorner, brcorner, color, line_thickness)
    return img


def crop(img, points):
    '''rogne l'image'''
    cropped_imgs = []
    for [(x0, y0), (x1, y1)] in points:
        cropped_img = img[y0:y1, x0:x1]
        cropped_img = resize_image(cropped_img, (224, 224))
        cropped_imgs.append(cropped_img)
    return cropped_imgs
