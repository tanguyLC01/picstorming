## Objectif (MVP): Un système de vérification du port du masque sur une vidéo

Alexandre LEMAIRE Marie CUBAYNES Hugo PEETERS Tanguy LECLOIREC Clara VERRET Augustin DE MONTBEL

# Fonctionnalité 1 : Installation du socle technique

 * installer OpenCV
 ```python
 python3 -m pip install opencv-contrib-python
```
 * Le fichier requirements.txt liste les différents modules nécessaires

# Fonctionnalité 2 : Effectuer un traitement sur une image et afficher le résultat du traitement

 * Pour chaque traitement d'une image, on crée la fonction qui réalise le traitement souhaité ainsi qu'une fonction test qui teste le bon fonctionnement de la fonction.

 
 * _Itération 1 : Appliquer un redimensionnement à une image :_ La fonction resize_image prend en argument une matrice de triplet représentant une image ainsi que le couple des nouvelles dimensions souhaitées et renvoie la matrice sous les nouvelles dimensions
 ```python
 from image_processing_operations import resize_image
 from test_resize_image import test_resize_image

 test_resize_image()
```
 * _Itération 2 : Dessiner un rectangle autour d'une zone d'intérêt :_ La fonction draw_rectangle dessine un rectangle de la couleur indiquée en argument autour de la zone délimitée par le coin haut à gauche et le coin bas à droite sur une image représentée par une matrice de triplet
```python
 from image_processing_operations import draw_rectangle
 from test_rectangle import test_rectangle
```
 test_rectangle()

  * _Itération 3 : rogner une image autour d'une zone d'intérêt :_ La fonction crop prend en entrée une image et une liste de liste de tuples (c'est à dire plusieurs rectangles). Elle renvoie une liste d'image rognée.
 ```python
 from image_processing_operation import crop
 from test_crop import crop
 ```


# Fonctionalité 3 : Converture des tests

 * Pour s'assurer que jusqu'ici le code écrit est juste, on réalise l'exécution de tous les tests. Pour cela, on télécharge le module coverage.
 ```python
 pip install coverage
 pip install pytest-cov
 ```
 * Dans bash entrer la commande
```python
 pytest --cov=picstorming --cov-report html ./Test/test/test_*
```

# Fonctionnalité 4 : Structuration de la base de donnée de test

 * Nous avons créé une base de donnée pour réaliser des tests. Des photos de personnes avec et sans masque se trouvent dans le répertoire Dataset dans les sous-répertoires with_mask et without_mask.

 * la fonction update_cvs permet à partir de toutes ces photos de construire un tableau test_data_info. Ainsi pour renforcer la base de donnée, il suffit d'ajouter des photos dans Test.data_test puis de lancer la fonction update_cvs. Les nouvelles photos vont s'afficher et il faut indiquer manuellement le nombre de personnes et si elles portent ou non un masque. Par exemple, si on a une photo de 2 personnes portant des masques on écrit : 2, no_mask, no_mask.
```python
 python -m Test.update_cvs
```

# Fonctionnalité 5 : Détecter un visage

 * Pour détecter les visages, nous avons téléchargé un modèle. Voivi le lien du modèle :

 https://github.com/sr6033/face-detection-with-OpenCV-and-DNN


 * La fonction detect_face2 renvoie une liste de couples de points caractérisant les rectangles autour des différents visages.
 ```python
 from utils_cv.facedetection import detect_face2
 ```

# Fonctionnalité 6 : Détecter un masque
 
 * Pour détecter les masques, nous avons téléchargé un modèle. Voici le lien du modèle :

 https://github.com/chandrikadeb7/Face-Mask-Detection

 * La fonction is_masked2 prend en argument une liste de tableaux représentant les images des visages rognées et le modèle qui détecte les masques. Elle renvoie un tuple : par exemple (True,0.95) signifie que la personne porte son masque avec une probabilité de 0,95.
 ```python
 from utils_cv.facedetection import is_masked2
 ```

# Fonctionnalité 7 : Processus de détection final

 * La fonction detection_process2 synthétise le programme en utilisant les fonctions is_masked et detect_face2. Elle prend en argument un tableau représentant une image ainsi ques les modèles chargés de reconnaître les visages dans l'image et de détecter si les visages portent un masque ou non. Elle retourne un tableau représentant l'image avec les visages encadrés en vert s'ils portent un masque et encadrés en rouge sinon. Le pourcentage de certitude du modèle ainsi que la mention 'mask' ou 'no mask' sont de plus indiqués au dessus de chaque visage.
 ```python
 from utils_cv.facedetection import detection_process2
 ```

# Fonctionnalité 8 : Interface graphique 

 * Pour faciliter le lancement du programme, on crée une interface graphique dans le fichier main.py qu'il suffira d'exécuter pour ensuite interagir avec l'interface graphique et lancer le programme. Pour démarrer l'interface graphique et le programme, entrer dans Bash à la racine du répertoire :
```python
python main.py
```
* Un fichier audio a enfin été ajouté afin de signaler aux personnes ne portant pas de masque 
# Fonctionnalité 9 : Tester la reconnaissance sur la base de données :

 * Le fonction test_model vérifie le bon fonctionnement du programme à partir de la base de donnée de la fonctionnalité 4
 ```python
 from test.test_model.test_model import test_model
 ```
 * La fonction renvoie :

 Validation loss: 0.04072757437825203, Validation accuracy: 0.9729729890823364




