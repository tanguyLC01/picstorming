from utils_cv.image_processing_operations import resize_image
import cv2

# fonction qui vérifie que l'image redimensionnée en 200x200 est bien de cette dimension après application de resize_image


def test_resize_image():
    # given
    img = cv2.imread(
        r'.\Test\test\data_images\mer.jpg')
    (h, w, d) = img.shape
    new_dim = (200, 200)
    # when
    resized_image = resize_image(img, new_dim)
    # then
    assert resized_image.shape == (new_dim[0], new_dim[1], d)


test_resize_image()
