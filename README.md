# Face recognition algorithms   

:busts_in_silhouette: This project was part of the assignments for [Biometrics System Concepts course](https://onderwijsaanbod.kuleuven.be/syllabi/e/H02C7AE.htm#activetab=doelstellingen_idp20969472) @ KU Leuven, for the academic year 2018/2019. The `localmodules` folder and parts of the algorithms were provided during the course.

Face recognition concernes the identifing or verifying a person's identity from an image or a video frame. It can be formulated as a classification problem, where the inputs are the images and the outputs are the identities of the people. This project analyses 3 traditional computer vision techniques for feature extraction in the context of face recognition:

1. PCA (Principal Components Analysis)
2. LDA (Linear Discirminant Analysis)
3. LBP (Local Binary Patterns)

## Data
The experiments are conducted using 3 different datasets:

1. AT&T Faces dataset
2. Labeled faces of the wild
3. Caltech Faces dataset

Use the methods defined in `localmodules/datasets.py` to extract the faces.

## Implementation
For each technique (PCA, LDA and LBP) we experiment with the 3 different datasets and we analyse an authentication and identification scenario. Firstly, we compute the similarity distances between two pairs of faces, by using the mentioned techniques for feature extraction from the image. If the distance is greater or equal to some predefined threshold, then the pair is labeled as a genuine, otherwise it is labeled as an impostor. Then, we look at the genuine impostor distributions and the F1 and accuracy scores for a range of different thresholds. We choose the most optimal threshold by analysing the plot of the F1 and accuracy scores. For the authentication scenarios we also report the Reciever Operating Characteristics (ROC) curves, whereas for the identification scenarios we report the Cumulative Matching Characteristics (CMC) curve. 

The implementation is entierly done in Python (also using Scikit-learn and OpenCV) and the main flow of the project is assembled in the Jupyter notebook `src/fingerprint-recognition.ipynb`.

## Overall evaluation
  1. Accuracy
  2. F1 score
  3. Averaged precision (AP) score
  4. Equal Error Rate (EER)
  5. Reciever Operating Characteristics (ROC) curve (TPR vs FPR)
  6. Area Under the curve (AUC) score
  7. Cumulative Matching Characteristics (CMC) curve


