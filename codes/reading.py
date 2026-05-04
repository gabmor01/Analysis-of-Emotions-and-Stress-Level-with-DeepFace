import os
import cv2
import matplotlib.pyplot as plt

# Percorso alla cartella
folder_path = "C:/Users/mesit/Downloads/DATASET/train/"

# Loop attraverso le cartelle
for emotion_folder in os.listdir(folder_path):
    emotion_path = os.path.join(folder_path, emotion_folder)

    if os.path.isdir(emotion_path):  # Controlla che sia una cartella
        for image_file in os.listdir(emotion_path):
            image_path = os.path.join(emotion_path, image_file)
            
            # Carica l'immagine
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Grayscale
            plt.imshow(img, cmap='gray')
            plt.title(f"Emotion: {emotion_folder}")
            plt.show()
            break  # Mostra solo la prima immagine
    break