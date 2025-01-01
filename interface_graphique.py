import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from Model.model import use_model
import tensorflow as tf
from utils_cv.facedetection import detection_process2, get_mask_status
import pygame

pygame.mixer.init()
pygame.mixer.music.load("Message audio.wav")
mask_model = tf.keras.models.load_model(
    "./ResNet50_v2/ResNet50_mask_detector.model")
face_model = cv2.dnn.readNet(
    "face_detector/deploy.prototxt", "face_detector/res10_300x300_ssd_iter_140000.caffemodel")


def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    try:
        frame = detection_process2(frame, mask_model, face_model)
        """ This part launch this audio if one of the persons is not wearing a mask"""
        if not(get_mask_status()):
            if not pygame.mixer.music.get_busy():  # Check if the music is already launched
                pygame.mixer.music.play(False)  # False: Run asynchronously
        else:
            pygame.mixer.music.pause()
            pygame.mixer.music.stop()
    except cv2.error as e:
        print(e)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    limage.imgtk = imgtk
    limage.configure(image=imgtk)
    limage.after(1, show_frame)


window = tk.Tk()  # Makes main window
window.wm_title("Digital Microscope")
window.config(background="#FFFFFF")

# Graphics window
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)


limage = tk.Label(imageFrame)
limage.grid(row=1, column=0)
cap = cv2.VideoCapture(0)

bouton = tk.Button(imageFrame, text="Lancer le programme",
                   command=show_frame)
bouton.grid(row=0, column=0)
photo = tk.PhotoImage(file="codingweeks2.png")
limage.imgtk = photo
limage.configure(image=photo)


window.mainloop()  # Starts GUI
