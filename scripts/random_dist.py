#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# ORT analysis - script random distributor - developed by Victor Ibañez
# 03.04.2021
# -------------------------------------------------------------------------------------


# -----------------------------------------------------
# import libraries
# -----------------------------------------------------  

import os
import cv2
import numpy as np
import random
from sklearn.cluster import KMeans

# -----------------------------------------------------
# apply kmeans and randomly choose videos
# ----------------------------------------------------- 

def random_dist(vids, nk):

	image_array = []

	for i in vids:
	    #video = os.path.join(os.path.dirname(i),i)
	    vidcap = cv2.VideoCapture(i)
	    success,image = vidcap.read()
	    cnt = 1

	    while success:

	        if cnt == 50:
	            img = image[:,:,0]
	            img = np.concatenate(img)
	            image_array.append(img)

	        elif cnt > 50:
	            break

	        success,image = vidcap.read()
	        cnt += 1

	from sklearn.cluster import KMeans
	kmeans = KMeans(n_clusters=nk, random_state=0).fit(image_array)

	t = set(kmeans.labels_)

	dic ={}
	for i,j in zip(vids, kmeans.labels_):
	    for k in t:
	        if k == j:
	            if j in dic:
	                dic[j].append(i)
	            else:
	                dic[j] = [i]

	selection = []

	for i in dic:
	    selection.append(random.choice(dic[i]))

	return selection
