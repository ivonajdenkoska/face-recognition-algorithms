# Face recognition algorithms

This project was part of the assignments for the [Biometrics System Concepts course](https://onderwijsaanbod.kuleuven.be/syllabi/e/H02C7AE.htm#activetab=doelstellingen_idp20969472) @ KU Leuven, for the academic year 2018/2019. Parts of the code and the algorithms were provided during the course.

Face recognition concernes the identifing or verifying a person's identity from an image or a video frame. It can be formulated as a classification problem, where the inputs are the images and the outputs are the identities of the people. This project analyses 3 traditional computer vision techniques for feature extraction in the context of face recognition:

1. PCA (Principal Components Analysis)
2. LDA (Linear Discirminant Analysis)
3. LBP (Local Binary Patterns)

## Data
The experiments are conducted using 3 different datasets:

1. AT&T Faces dataset
2. Labeled faces of the wild
3. Caltech Faces dataset

## Implementation
Firstly, we compute the similarity distances between two pairs of faces, by using the mentioned techniques for feature extraction. If the distance is greater or equal to some predefined threshold, then the pair is labeled as a genuine, otherwise it is labeled as an impostor. We look at the genuine impostor distributions, the F1 and accuracy scores for a range of different thresholds etc.

Furthermore, for each technique we experiment with the 3 different datasets and we analyse an authentication and identification scenario. For the authentication scenarios we report the Reciever Operating Characteristics (ROC) curves, whereas for the identification scenarios we report the Cumulative Matching Characteristics (CMC) curve. 
