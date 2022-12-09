#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# ORT analysis - script raw data - developed by Victor Ibañez
# 03.04.2021
# -------------------------------------------------------------------------------------


# -----------------------------------------------------
# import libraries
# -----------------------------------------------------

import glob, os
import cv2

# -----------------------------------------------------
# create raw data
# -----------------------------------------------------

def create_raw_data(video, path, dic):

	# append label video frames to list
	vidcap = cv2.VideoCapture(video)
	success,image = vidcap.read()
	count = 0
	l = []

	while success:
	    res = cv2.resize(image, dsize=(150, 150), interpolation=cv2.INTER_CUBIC)
	    l.append(res)  
	    success,image = vidcap.read()
	    count += 1

	exp = []

	# compare to object list and create data
	o_list = list(dic.values())

	for i in range(len(o_list)):
	    target_name = list(dic.keys())[i]
	    target_path = os.path.join(path,target_name)

	    for j in range(0,len(o_list[i]),2):
	        for frame,count in zip((l[o_list[i][j]:o_list[i][j+1]+1]),range(o_list[i][j],o_list[i][j+1]+1)):
	            cv2.imwrite(os.path.join(target_path,target_name) + "_%d.jpg" % count, frame)
	            exp.append(count)
	            
	for i in range(0,len(l)):
	    if i not in exp:
	        cv2.imwrite(os.path.join(path,'no','no') + "_%d.jpg" % i, l[i])
        
