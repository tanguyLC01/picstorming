import csv
import cv2
import os
from utils_cv.image_processing_operations import resize_image


def photo_in_database(filename):
    '''renvoie un booléen pour savoir si la photo à ajouter est déjà dans la database qui entraine le modèle'''
    with open("./test/test_data_info.csv", "r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for rows in csv_reader:
            if filename in rows.values():
                return True
    return False


rows_list = []
for file_name in os.listdir("./Test/data_test"):
    if not photo_in_database(file_name): 
        #la photo n'est pas déjà dans la database, on l'ajoute
        img = cv2.imread("./Test/data_test/" + file_name)
        img = resize_image(img, (224, 224))
        cv2.imshow("image", img) # on affiche l'image pour que l'utilisateur rentre ses caractéristiques
        cv2.waitKey(0)
        cv2.destroyAllWindows
        num_person, type_photo = input(
            "Entre le nombre de personne et le type(ex: 2 no_mask,mask): ").split(" ")
        num_person = int(num_person)
        rows_list.append([file_name, num_person, type_photo])

with open("./Test/test_data_info.csv", "a") as file:
    csv_writer = csv.writer(file)
    for row in rows_list:
        csv_writer.writerow(row) # on ajoute la nouvelle image dans test_data_info.csv
