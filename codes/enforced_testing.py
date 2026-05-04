import os
from deepface import DeepFace
import cv2
import csv
import shutil

def calculate_stress(emotion_probs):
    # Estrapola le percentuali
    happy = emotion_probs.get("happy", 0)
    sad = emotion_probs.get("sad", 0)
    neutral = emotion_probs.get("neutral", 0)
    anger = emotion_probs.get("angry", 0)
    fear = emotion_probs.get("fear", 0)
    disgust = emotion_probs.get("disgust", 0)
    surprise = emotion_probs.get("surprise", 0)

    # Calcola il livello di stress
    if happy > 30 or neutral > 50:
        return "Low Stress"
    elif fear + sad + surprise > 90:
        return "Medium Stress"
    elif anger + disgust > 40:
        return "High Stress"
    else:
        return "Undefined Stress"

# Imposta il percorso della cartella
folder_path = "val_set/not detected/images"
emotion_error_folder = "val_set/not detected/errori vari"
stress_results_folder = "val_set/not detected/stress_results"

# Loop attraverso le immagini nella cartella
for emotion_folder in os.listdir(folder_path):
    emotion_path = os.path.join(folder_path, emotion_folder)

    # Creazione della sottocartella "errori vari" per ciascuna emozione
    emotion_error_emotion = os.path.join(emotion_error_folder, emotion_folder)
    os.makedirs(emotion_error_emotion, exist_ok=True)

    with open(f"{stress_results_folder}/not_detected_stress_results_{emotion_folder}.csv", "w", newline="") as csvfile:
        fieldnames = ["file_name", "dominant_emotion", "emotions", "stress_level"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for file_name in os.listdir(emotion_path):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Controlla il formato
                image_path = os.path.join(emotion_path, file_name)

                # Carica l'immagine
                img = cv2.imread(image_path)

                print(f"File: {file_name}")

                # Rileva le emozioni
                emotions = DeepFace.analyze(img, ['emotion'], enforce_detection = False)[0]

                if(emotions['dominant_emotion'] != emotion_folder): 
                    shutil.copy(image_path, os.path.join(emotion_error_emotion, file_name))

                stress_level = calculate_stress(emotions['emotion'])
                print(f"Stress Level: {stress_level}")

                writer.writerow({
                    "file_name": file_name,
                    "dominant_emotion": emotions['dominant_emotion'],
                    "emotions": {key: round(value, 3) for key, value in emotions['emotion'].items()},
                    "stress_level": stress_level
                })