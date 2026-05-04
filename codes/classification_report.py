import os
import csv
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Percorso alla cartella contenente i CSV
csv_folder = "val_set/stress_results"

# Variabili per raccogliere etichette vere e predette
true_labels = []
predicted_labels = []

# Loop attraverso i file CSV
for csv_file in os.listdir(csv_folder):
    if csv_file.endswith(".csv"):
        # Determina l'etichetta vera dalla struttura del nome del file
        true_emotion = csv_file.split("_")[-1].split(".")[0]  # Es. da "stress_results_happy.csv" estraiamo "happy"
        
        # Legge il file CSV
        with open(os.path.join(csv_folder, csv_file), "r") as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                file_name = row["file_name"]
                predicted_emotion = row["dominant_emotion"]
                
                true_labels.append(true_emotion)
                predicted_labels.append(predicted_emotion)

# Calcolo delle metriche
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average="weighted")
recall = recall_score(true_labels, predicted_labels, average="weighted")
f1 = f1_score(true_labels, predicted_labels, average="weighted")

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Report dettagliato
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels))

# Confusion Matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=np.unique(true_labels))

# Visualizzazione della Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()