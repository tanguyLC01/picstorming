from utils_cv.image_processing_operations import crop
import cv2
import numpy as np

# fonction qui vérifie que l'image rognée puis redimensionnée en 224x224 est différente de l'image initiale


def test_crop():
    # given
    img = cv2.imread(
        r'.\Test\test\data_images\mer.jpg')
    # la fonction 'crop' redimensionne l'image en 224x224 donc pour vérifier que les imges sont différentes on redimensionne l'image initiale
    new_img = cv2.resize(img, (224, 224))
    points = [[(5, 25), (50, 100)]]
    # when
    cropped_image = crop(new_img, points)
    # then
    assert np.array_equal(new_img, cropped_image) == False


test_crop()
