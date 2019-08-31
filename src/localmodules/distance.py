#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:20:17 2019

@author: dvdm
"""

import numpy as np
import matplotlib.pyplot as plt

def L2_distance(emb1, emb2):
    return np.sqrt(np.sum(np.square(emb1 - emb2)))

# kullback_leibler_divergence
def KLD(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / (q[filt]+ 1.0e-7)))

def CHI2(histA, histB, eps=1e-10):
	# compute the chi-squared distance
	d = 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))
 
	# return the chi-squared distance
	return d

def show_pair(faces, embedded_faces, idx1, idx2):
    plt.figure(figsize=(8,3))
    plt.suptitle(f'Distance = {L2_distance(embedded_faces[idx1], embedded_faces[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(faces.images[idx1],cmap='gray')
    plt.subplot(122)
    plt.imshow(faces.images[idx2],cmap='gray');    

