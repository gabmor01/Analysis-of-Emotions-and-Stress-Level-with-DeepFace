## Project Overview

This project analyzes human facial images to estimate stress levels based on expressed emotions. The images are sourced from the AffectNet and RAF-DB datasets and processed using the DeepFace framework for emotion recognition.

An indicator is developed to classify stress into three levels (low, medium, and high) based on detected emotions. The approach is evaluated by assessing DeepFace’s ability to correctly recognize and distinguish facial expressions.

Results show that while DeepFace performs well on certain emotions, it struggles with similar emotional states, impacting stress classification accuracy. Despite these limitations, the method demonstrates potential for stress monitoring in real-world applications, such as detecting fatigue or risky behavior. Future improvements include dataset optimization and integration with more advanced emotion recognition models.



## Project structure

codes/ → Contains the codes used for the analysis: 
  - reading.py → checks that the dataset to be analyzed is complete and well-structured
  - testing_entire_dataset.py → runs the complete analysis of the chosen dataset and produces the stress results
  - enforced_testing.py → tests the entire dataset in enforced mode
  - classification_report.py → analyzes the produced stress results, evaluates the model, and produces the confusion matrix

raf_db/ or val_set/ → Contains all the data and analyses produced for each dataset (mirrored folders)
  - analisi_db_totale.txt or analisi_val_set.txt → text file that shows all collected data
  - Analisi RAF_DB.xlsx or Analisi Val_Set.xlsx → Excel file that shows the produced graphs
  - confusion_matrix_raf_db.png or confusion_matrix_val_set.png → the confusion matrix produced by the analysis
  - stress_results/ → folder containing CSV files for each classified emotion
  - not detected/ → folder containing the stress results and the confusion matrix of the analysis on the not detected images

relazioni/ → Contains all documents produced by the project
  - Analisi delle Emozioni e del Livello di Stress con DeepFace.pdf → Final report with methodology, results, and discussion.
  - Analysis of Emotions and Stress Level with DeepFace → Final report in English
  - presentazione.pptx → Support slides for the project presentation.
