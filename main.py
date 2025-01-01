import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from Model.model import use_model
import tensorflow as tf
from utils_cv.facedetection import detection_process2, get_mask_status
import pygame


def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # Try mode because cv2 resize bugs frequently
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
    # Adding frame video to the image object
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    limage.imgtk = imgtk
    limage.configure(image=imgtk)
    limage.after(1, show_frame)


if __name__ == "__main__":
    # Initalizing music pygame and loading music file
    pygame.mixer.init()
    pygame.mixer.music.load("Message audio.wav")

    # Load models for mask detection and face detection
    mask_model = tf.keras.models.load_model(
        "./ResNet50_v2/ResNet50_mask_detector.model")
    face_model = cv2.dnn.readNet(
        "face_detector/deploy.prototxt", "face_detector/res10_300x300_ssd_iter_140000.caffemodel")
    # Creates main window
    window = tk.Tk()
    window.wm_title("Détection de masque")
    window.config(background="#FFFFFF")

    # Graphics window
    imageFrame = tk.Frame(window, width=600, height=500)
    imageFrame.grid(row=0, column=0, padx=10, pady=2)

    # Add video container
    limage = tk.Label(imageFrame)
    limage.grid(row=1, column=0)
    cap = cv2.VideoCapture(0)

    # Add button
    bouton = tk.Button(imageFrame, text="Lancer le programme",
                       command=show_frame)
    bouton.grid(row=0, column=0)
    photo = tk.PhotoImage(file="codingweeks2.png")
    limage.imgtk = photo
    limage.configure(image=photo)

    window.mainloop()  # Starts GUI
