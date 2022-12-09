#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
# ORT analysis - script raw data correct - developed by Victor Ibañez
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

def create_raw_data_corr(video, path, dic, cnt, ref_point, time, iteration):

	# append label video frames to list
	vidcap = cv2.VideoCapture(video)
	success,image = vidcap.read()
	count = 1
	start = 15
	fps =  vidcap.get(cv2.CAP_PROP_FPS)
	max_time = (int(time)*60*int(fps))+start
	l = []


	while success:

		res1 = image[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]] # crop frame
		res = cv2.resize(res1, dsize=(150, 150), interpolation=cv2.INTER_CUBIC) # resize

		if count >= start and count < max_time:
			l.append(res)

		elif count > max_time:
			break

		success,image = vidcap.read()
		count += 1

	# compare to object list and create data
	o_list = list(dic.values())

	for i in range(len(o_list)):
		target_name = list(dic.keys())[i]
		target_path = os.path.join(path,target_name)
		for j in range(0,len(o_list[i]),2):
		    for frame,count in zip((l[o_list[i][j]:o_list[i][j+1]+1]),range(o_list[i][j],o_list[i][j+1]+1)):
		        cv2.imwrite(os.path.join(target_path,target_name) + "_corr_{0}_{1}_{2}.jpg".format(cnt, count, iteration), frame)

        
