"""
@author: dvdm
"""

# import the necessary packages
from sklearn.datasets.base import Bunch
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces

from scipy import io
import numpy as np
import cv2
import copy

# limit the samples to max number of targets (to limit computation times)
def limit_samples(faces_in, max_targets):
    faces_out = copy.deepcopy(faces_in)
    if max_targets is not None:
        n_samples = faces_in.images.shape[0]
        targets = faces_in.target
        labels = np.unique(targets)
        n_targets = labels.shape[0]
        face_ids = np.random.randint(n_targets, size=min(n_targets, max_targets))
        sample_ids = [sample_id for sample_id in range(n_samples) if targets[sample_id] in labels[face_ids]]
        faces_out.data = faces_in.data[sample_ids,]
        faces_out.images = faces_in.images[sample_ids,]
        faces_out.target = faces_in.target[sample_ids,]
    return faces_out

def load_caltech_people(datasetPath, min_faces=10, face_size=(47, 62)):
# load the CALTECH faces dataset
# this includes: reading the raw data; reading bounding box data; extracting faces and resizing, flattening;
# checking on balance (minimal nr of faces/individual, equal number)
    
    # grab in all the subdirs all the image paths associated with the faces
    imagePaths = datasetPath.rglob("*.jpg")
    
    # then load the bounding box data stored in a Matlab .mat file
    bbData = io.loadmat(datasetPath.joinpath("ImageData.mat"))
    bbData = bbData["SubDir_Data"].T

    # set the random seed, then initialize the data matrix and labels
    images = []
    data = []
    labels = []

    # loop over the image paths
    for imagePath in imagePaths:
        # load the image and convert it to grayscale
        image = cv2.imread(str(imagePath))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # grab the bounding box associated with the current image, extract the face
        # ROI, and resize it to a canonical size
        imagePathStem = str(imagePath.stem)
        k = int(imagePathStem[imagePathStem.rfind("_") + 1:][:4]) - 1
        (xBL, yBL, xTL, yTL, xTR, yTR, xBR, yBR) = bbData[k].astype("int")
        face = gray[yTL:yBR, xTL:xBR]
        face = cv2.resize(face, face_size)

        # update the data matrix and associated labels
        images.append(face)
        face_flatten = face.flatten()
        data.append(face_flatten)
        labels.append(imagePath.parent.name)

    # convert the data matrix and labels list to a NumPy array
    images = np.array(images)
    data = np.array(data)
    labels = np.array(labels)
        
    return Bunch(data=data, images=images, target=labels)
   
def load_lfw_faces(dataset, min_faces, max_persons):
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces, resize=0.4)
    return limit_samples(lfw_people, max_persons)
    
def load_ATT_faces(dataset, min_faces, max_persons):
    att_faces = fetch_olivetti_faces()
    return limit_samples(att_faces, max_persons)

def load_caltech_faces(dataset, min_faces, max_persons):
    caltech_people = load_caltech_people(datasetPath=dataset, min_faces=min_faces, face_size=(47, 62))
    return limit_samples(caltech_people, max_persons)
    
def load_faces(facesDB, dataset, min_faces=10, max_targets=20):
    loader = {
        "ATT": load_ATT_faces,
        "CALTECH": load_caltech_faces,
        "LFW": load_lfw_faces
    }
    return loader[facesDB](dataset, min_faces, max_targets)